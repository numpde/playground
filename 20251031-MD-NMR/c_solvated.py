# /home/ra/repos/playground/20251031-MD-NMR/c_solvated.py

"""
c_solvated.py

Build and simulate an explicit-solvent box for each
protonation/tautomer state defined in a_init.TARGETS.

Pipeline context:
- a_init.py writes per-state starting structures
  (e.g. aspirin_neutral_init.pdb) with the correct
  protonation for a given solvent / pH scenario.
- b_openmm.py sanity-checks each state in vacuum.
- c_solvated.py (this file) puts that state into
  bulk solvent, equilibrates it, and runs short
  production MD so later steps can cluster
  conformers and feed them to NMR prediction.

What this script actually does:
1. Load the *_init.pdb for each TARGETS entry and
   rebuild the OpenFF Molecule with that geometry.
   The solute is treated as a single residue "LIG".
2. Pack a periodic cubic box (BOX_EDGE_NM) with
   explicit solvent molecules ("SOL") at roughly
   experimental molar density. Solvent choices:
   water, DMSO, CDCl3. Each solvent copy is given
   a random rotation/orientation before placement,
   not just tiled.
3. Build an OpenMM System using OpenFF 2.x (and
   TIP3P for water), add a MonteCarloBarostat
   (1 bar, 300 K), and a LangevinMiddleIntegrator
   (300 K, 1 fs timestep).
4. Add a CustomExternalForce that applies a
   harmonic positional restraint (k_restraint) to
   heavy atoms of the solute only. These restraints
   are active during PRE_EQ to let solvent resolve
   clashes and form a first solvation shell without
   letting the solute deform.
5. PRE_EQ:
   - minimize
   - short restrained NPT run (PRE_EQ_PS)
   - checkpoint
6. After PRE_EQ, set k_restraint → 0. Then run a
   longer unrestrained NPT equilibration (EQ_PS),
   checkpoint again, and dump a wrapped PDB
   snapshot with residues imaged back into the
   primary unit cell.
7. Production:
   - continue the same Simulation
   - run geometrically increasing chunks
     (1 ps, 10 ps, 100 ps, 1000 ps)
   - stream thermodynamics
   - append all coordinates to a single DCD
   - write checkpoint PDBs after each chunk

Outputs (per target, per solvent):
- <name>_<solvent>_minimized.pdb
- <name>_<solvent>_eq.pdb
- <name>_<solvent>.dcd
- <name>_<solvent>.log
- periodic checkpoints like
  <name>_<solvent>_chk_100ps.pdb
plus PRE_EQ/EQ checkpoint blobs (.chk).

Downstream expectations:
- d_extract_solute.py can now trivially isolate
  the solute as residue "LIG" and ignore solvent
  "SOL".
- e_cluster.py can cluster heavy-atom RMSD across
  production frames, estimate conformer basins,
  and give representative structures for QM/NMR.

This script assumes CPU platform by default and
uses all local threads. GPU migration later is
mechanical: same topology, same forces, just a
different Platform().
"""

import math
import os
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import openmm
import openmm.unit as unit
from bugs import mkdir
from openff.toolkit import Molecule, Topology, ForceField
from openff.units import unit as offunit
from openmm.app import (
    Simulation,
    DCDReporter,
    StateDataReporter,
    PDBFile,
)
from rdkit import Chem
from rdkit.Chem import AllChem

import a_init
from a_init import TARGETS  # [{'name','smiles','pdb','charge',...}, ...]

# ---------------------------------------------------------------------
# output locations
# ---------------------------------------------------------------------

A_INIT_DIR = Path(a_init.__file__).with_suffix('')  # where a_init wrote *_init.pdb
OUT_DIR = mkdir(Path(__file__).with_suffix(''))  # ./c_solvated/

# ---------------------------------------------------------------------
# MD / thermodynamic settings
# ---------------------------------------------------------------------

SOLVENT_CHOICE = "dmso"  # "dmso", "water", "cdcl3"
BOX_EDGE_NM = 3.0  # cubic periodic box edge length (nm)

TARGET_TEMP = 300 * unit.kelvin
TARGET_PRESSURE = 1 * unit.bar

TIMESTEP_PS = 0.001  # 1 fs = 0.001 ps
FRICTION = 1.0 / unit.picosecond

PRE_EQ_PS = 50.0  # short restrained pre-equilibration
EQ_PS = 950.0  # long NPT settle
REPORT_INT_STEPS = 1000  # report every 1000 steps (~1 ps at 1 fs)

PRE_EQ_STEPS = int(PRE_EQ_PS / TIMESTEP_PS)  # 50,000 steps
EQ_STEPS = int(EQ_PS / TIMESTEP_PS)  # 950,000 steps

# Harmonic positional restraint strength for solute heavy atoms during PRE_EQ.
# Units: kJ/mol/nm^2. Will be turned off (set to 0) after PRE_EQ.
PRE_EQ_RESTRAINT_K = 1000.0 * unit.kilojoule_per_mole / (unit.nanometer ** 2)


def _load_coords_from_init_pdb(pdb_path: Path) -> Tuple[np.ndarray, PDBFile]:
    """
    Read the *_init.pdb from a_init for this protomer.
    Returns:
      coords_A  : (N,3) float64 in Å
      pdb_obj   : OpenMM PDBFile (has topology)
    Note: PDBFile gives positions in nm; convert to Å.
    """
    pdb_obj = PDBFile(str(pdb_path))
    pos_nm = pdb_obj.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # (N,3)
    coords_A = pos_nm * 10.0  # 1 nm = 10 Å
    return (coords_A, pdb_obj)


def _rdkit_from_smiles_with_H(smiles: str) -> Chem.Mol:
    """
    Rebuild RDKit mol from SMILES with explicit H, preserving atom order.
    The SMILES here is the same one used in a_init, so ordering should match
    the *_init.pdb atoms.
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    rdmol = Chem.AddHs(rdmol)
    return rdmol


def _off_mol_with_conformer(rdmol: Chem.Mol, coords_A: np.ndarray) -> Molecule:
    """
    RDKit mol -> OpenFF Molecule with same atom ordering and an attached conformer.
    coords_A: (N,3) Å
    """
    off_mol = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)
    off_mol.add_conformer(coords_A * offunit.angstrom)
    return off_mol


def _center_coords_nm(coords_nm: np.ndarray) -> np.ndarray:
    """
    Translate coords so centroid is ~0.
    coords_nm: (N,3) nm
    """
    ctr = coords_nm.mean(axis=0, keepdims=True)
    return coords_nm - ctr


def _make_box_vectors(box_nm: float) -> np.ndarray:
    """
    Create 3x3 box matrix for an orthorhombic cube of edge box_nm [nm].
    """
    a = box_nm
    return np.array(
        [
            [a, 0.0, 0.0],
            [0.0, a, 0.0],
            [0.0, 0.0, a],
        ],
        dtype=float,
    )


def _estimate_n_solvent(solvent_kind: str, box_nm: float) -> int:
    """
    Estimate how many solvent molecules fit in a box_nm^3 nm^3 at ~bulk density.

    Bulk molar densities (mol/L):
      water:   ~55.5
      dmso:    ~14.0   (ρ≈1.1 g/mL, M≈78)
      cdcl3:   ~12.4   (ρ≈1.48 g/mL, M≈119)

    molecules_per_nm3 = mol/L * 0.602214
    """
    if solvent_kind == "water":
        mol_per_L = 55.5
    elif solvent_kind == "dmso":
        mol_per_L = 14.0
    elif solvent_kind == "cdcl3":
        mol_per_L = 12.4
    else:
        raise ValueError(f"Unknown solvent_kind {solvent_kind}")

    molecules_per_nm3 = mol_per_L * 0.602214
    vol_nm3 = box_nm ** 3
    n = int(round(molecules_per_nm3 * vol_nm3))
    return max(n, 1)


def _embed_minimize_solvent_template(smiles: str, label: str) -> Tuple[np.ndarray, Molecule]:
    """
    Make a single solvent molecule template:
    - RDKit embed (ETKDGv3) + MMFF94s optimize
    - convert to OpenFF Molecule
    - return centered coords (nm) + off_mol
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Bad solvent SMILES for {label}")
    rdmol = Chem.AddHs(rdmol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 12345
    cid = AllChem.EmbedMolecule(rdmol, params)
    if cid < 0:
        raise RuntimeError(f"Conformer gen failed for solvent {label}")

    AllChem.MMFFOptimizeMolecule(
        rdmol,
        mmffVariant="MMFF94s",
        maxIters=200,
        confId=cid,
    )

    # pull coords in Å
    conf = rdmol.GetConformer(cid)
    Ns = rdmol.GetNumAtoms()
    coords_A = np.zeros((Ns, 3), dtype=float)
    for i in range(Ns):
        p = conf.GetAtomPosition(i)
        coords_A[i] = (p.x, p.y, p.z)

    # to nm and center
    coords_nm = coords_A * 0.1
    coords_nm = _center_coords_nm(coords_nm)

    off_solv = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)
    off_solv.add_conformer(coords_A * offunit.angstrom)

    return (coords_nm, off_solv)


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """
    Generate a random 3x3 rotation matrix (uniform over SO(3)).
    """
    # method: random unit quaternion
    u1, u2, u3 = rng.random(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    # quaternion -> rotation matrix
    R = np.array([
        [1 - 2 * (q3 * q3 + q4 * q4), 2 * (q2 * q3 - q4 * q1), 2 * (q2 * q4 + q3 * q1)],
        [2 * (q2 * q3 + q4 * q1), 1 - 2 * (q2 * q2 + q4 * q4), 2 * (q3 * q4 - q2 * q1)],
        [2 * (q2 * q4 - q3 * q1), 2 * (q3 * q4 + q2 * q1), 1 - 2 * (q2 * q2 + q3 * q3)],
    ], dtype=float)
    return R


def _place_solvent_grid_random_orient(
        template_coords_nm: np.ndarray,
        n_copies: int,
        box_nm: float,
        min_dist_from_origin_nm: float = 0.4,
        seed: int = 12345,
) -> List[np.ndarray]:
    """
    Place n_copies solvent molecules in a simple 3D lattice, each with a random
    rotation and translation. Skip cells too close to the solute center, then
    fill remaining if needed.

    Returns:
      coords_list_nm: list of (Ns,3) arrays for each solvent copy in nm
    """
    Ns = template_coords_nm.shape[0]

    # lattice centers
    n_per_dim = math.ceil(n_copies ** (1.0 / 3.0))
    capacity = n_per_dim ** 3
    if capacity < n_copies:
        n_per_dim += 1
        capacity = n_per_dim ** 3

    centers = []
    for ix in range(n_per_dim):
        for iy in range(n_per_dim):
            for iz in range(n_per_dim):
                cx = (ix + 0.5) / n_per_dim - 0.5
                cy = (iy + 0.5) / n_per_dim - 0.5
                cz = (iz + 0.5) / n_per_dim - 0.5
                centers.append((cx * box_nm, cy * box_nm, cz * box_nm))

    rng = np.random.default_rng(seed)
    rng.shuffle(centers)

    coords_list_nm: List[np.ndarray] = []
    used = 0

    def _rot_then_shift(R: np.ndarray, shift_nm: np.ndarray) -> np.ndarray:
        # template_coords_nm is (Ns,3)
        rotated = template_coords_nm @ R.T  # (Ns,3)
        return rotated + shift_nm[None, :]  # broadcast add

    # first pass: avoid solute core
    for (cx, cy, cz) in centers:
        if used >= n_copies:
            break
        dist0 = math.sqrt(cx * cx + cy * cy + cz * cz)
        if dist0 < min_dist_from_origin_nm:
            continue
        R = _random_rotation_matrix(rng)
        shift = np.array([cx, cy, cz], dtype=float)
        coords_list_nm.append(_rot_then_shift(R, shift))
        used += 1

    # fallback fill
    if used < n_copies:
        for (cx, cy, cz) in centers:
            if used >= n_copies:
                break
            R = _random_rotation_matrix(rng)
            shift = np.array([cx, cy, cz], dtype=float)
            coords_list_nm.append(_rot_then_shift(R, shift))
            used += 1

    return coords_list_nm


def _build_solvated_sim(
        job: Dict,
        solvent_kind: str,
        box_nm: float,
) -> Simulation:
    """
    Build an OpenMM Simulation with:
    - solute from a_init (correct protonation/tautomer)
    - explicit solvent box at approx bulk density
    - periodic boundary conditions
    - barostat at 1 bar / 300 K
    - Langevin thermostat at 300 K
    - harmonic restraints on solute heavy atoms via a CustomExternalForce
      (restraints active initially, will be turned off after PRE_EQ)

    Returns:
      Simulation (GPU if available via CUDA/HIP/OpenCL, else multithreaded CPU)
    """

    name = job["name"]
    smiles = job["smiles"]
    pdb_rel = job["pdb"]
    pdb_path = A_INIT_DIR / pdb_rel

    # --- solute coords from a_init pdb
    (solute_coords_A, _pdb_obj) = _load_coords_from_init_pdb(pdb_path)
    rdmol_solute = _rdkit_from_smiles_with_H(smiles)
    off_solute = _off_mol_with_conformer(rdmol_solute, solute_coords_A)

    # center solute in nm at origin
    solute_coords_nm = (solute_coords_A * 0.1)  # Å -> nm
    solute_coords_nm = _center_coords_nm(solute_coords_nm)

    n_solute_atoms = off_solute.n_atoms
    heavy_idx_solute: List[int] = [
        i for (i, atom) in enumerate(off_solute.atoms)
        if atom.atomic_number != 1
    ]

    # --- solvent template
    if solvent_kind == "water":
        solv_smiles = "O"  # H2O
    elif solvent_kind == "dmso":
        solv_smiles = "CS(=O)C"  # DMSO
    elif solvent_kind == "cdcl3":
        solv_smiles = "ClC(Cl)Cl"  # CHCl3 ~ CDCl3 (ignore isotope)
    else:
        raise ValueError(f"Unknown solvent {solvent_kind}")

    (solvent_template_nm, off_solvent) = _embed_minimize_solvent_template(
        solv_smiles,
        solvent_kind,
    )

    n_solvent = _estimate_n_solvent(solvent_kind, box_nm)

    solvent_coords_list_nm = _place_solvent_grid_random_orient(
        template_coords_nm=solvent_template_nm,
        n_copies=n_solvent,
        box_nm=box_nm,
        min_dist_from_origin_nm=0.4,
        seed=12345,
    )

    # --- stitch coordinates [solute first, then all solvent copies]
    coords_all_nm = np.concatenate(
        [solute_coords_nm] + solvent_coords_list_nm,
        axis=0,
    )  # (N_total,3) nm

    # --- build OpenFF Topology for solute + many solvent molecules
    molecules_for_top = [off_solute] + [off_solvent] * len(solvent_coords_list_nm)

    box_vectors = _make_box_vectors(box_nm)  # (3,3), nm
    top_off = Topology.from_molecules(molecules_for_top)
    top_off.box_vectors = box_vectors * offunit.nanometer

    # --- parameterize
    # water needs tip3p; other solvents use general OpenFF
    if solvent_kind == "water":
        ff = ForceField("openff-2.0.0.offxml", "tip3p.offxml")
    else:
        ff = ForceField("openff-2.0.0.offxml")

    system = ff.create_openmm_system(top_off)

    # add NPT barostat
    system.addForce(openmm.MonteCarloBarostat(TARGET_PRESSURE, TARGET_TEMP))

    # add harmonic positional restraints on heavy solute atoms
    # We'll control strength via global parameter k_restraint
    restraint = openmm.CustomExternalForce(
        "0.5*k_restraint*((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    restraint.addGlobalParameter(
        "k_restraint",
        PRE_EQ_RESTRAINT_K,
    )
    restraint.addPerParticleParameter("x0")
    restraint.addPerParticleParameter("y0")
    restraint.addPerParticleParameter("z0")

    for idx in heavy_idx_solute:
        ref_pos = coords_all_nm[idx]  # nm
        restraint.addParticle(
            idx,
            [ref_pos[0], ref_pos[1], ref_pos[2]],
        )

    system.addForce(restraint)

    # integrator
    integrator = openmm.LangevinMiddleIntegrator(
        TARGET_TEMP,
        FRICTION,
        TIMESTEP_PS * unit.picoseconds,
    )

    # --- choose best available OpenMM platform
    platform = None
    platform_name = None
    for cand in ("CUDA", "HIP", "OpenCL", "CPU"):
        try:
            platform = openmm.Platform.getPlatformByName(cand)
            platform_name = cand
            break
        except Exception:
            continue

    if platform is None:
        raise RuntimeError("No valid OpenMM platform found (CUDA/HIP/OpenCL/CPU).")

    # platform-specific properties
    if platform_name in ("CUDA", "HIP"):
        # mixed precision = good balance of speed/stability
        platform_props = {"Precision": "mixed"}
        print(f"[info] using {platform_name} platform (GPU, mixed precision)")
    elif platform_name == "OpenCL":
        # 'single' is widely supported across OpenCL devices
        platform_props = {"Precision": "single"}
        print(f"[info] using {platform_name} platform (accelerator/OpenCL, single precision)")
    elif platform_name == "CPU":
        n_threads = os.cpu_count() or 1
        platform_props = {"Threads": str(n_threads)}
        print(f"[info] using {platform_name} platform with {n_threads} threads")
    else:
        # Fallback safety branch; usually not hit
        platform_props = {}
        print(f"[info] using {platform_name} platform (no special properties)")

    # Simulation
    top_omm = top_off.to_openmm()

    # rename residues for downstream extraction:
    # first residue = solute
    residues = list(top_omm.residues())
    if residues:
        residues[0].name = "LIG"
        for res in residues[1:]:
            res.name = "SOL"

    sim = Simulation(top_omm, system, integrator, platform, platform_props)

    # set positions and periodic box
    sim.context.setPositions(coords_all_nm * unit.nanometer)
    a_vec = box_vectors[0] * unit.nanometer
    b_vec = box_vectors[1] * unit.nanometer
    c_vec = box_vectors[2] * unit.nanometer
    sim.context.setPeriodicBoxVectors(a_vec, b_vec, c_vec)

    # By default PRE_EQ_RESTRAINT_K is active.
    # We'll later set it to 0 after PRE_EQ via:
    # sim.context.setParameter("k_restraint", 0*unit.kilojoule_per_mole/unit.nanometer**2)
    return sim


def _write_snapshot_pdb_wrapped(sim: Simulation, out_path: Path) -> None:
    """
    Wrap each residue into the primary unit cell for visualization:
    keep residue intact, shift COM into [0,L) in each dimension,
    then write a PDB with those wrapped coords.
    Assumes orthorhombic box.
    """
    state = sim.context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # (N,3)

    (avec, bvec, cvec) = sim.context.getState().getPeriodicBoxVectors(asNumpy=True)

    box_lengths = np.array([
        avec[0].value_in_unit(unit.nanometer),
        bvec[1].value_in_unit(unit.nanometer),
        cvec[2].value_in_unit(unit.nanometer),
    ], dtype=float)  # (3,)

    wrapped_nm = np.zeros_like(pos_nm)
    top = sim.topology

    for res in top.residues():
        idxs = [atom.index for atom in res.atoms()]
        coords = pos_nm[idxs, :]
        center = coords.mean(axis=0)

        center_wrapped = center - np.floor(center / box_lengths) * box_lengths
        shift = center_wrapped - center
        wrapped_nm[idxs, :] = coords + shift

    wrapped_q = wrapped_nm * unit.nanometer
    with open(out_path, "w") as fh:
        PDBFile.writeFile(top, wrapped_q, fh, keepIds=True)


def run_production_in_chunks(
        sim: Simulation,
        name: str,
        solvent_choice: str,
        base_ps: float = 1.0,
        multipliers: List[float] = [1.0, 10.0, 100.0, 1000.0],
        timestep_ps: float = TIMESTEP_PS,
        report_interval_steps: int = REPORT_INT_STEPS,
        dcd_path: Optional[Path] = None,
        log_path: Optional[Path] = None,
):
    """
    After equilibration, run production in geometrically increasing chunks
    (1 ps, 10 ps, 100 ps, 1000 ps, ...) and:
    - append to a single DCD
    - log thermodynamics
    - write a checkpoint PDB after each chunk
    """
    import sys

    sim.reporters = [
        None,  # chunk-scoped live reporter (we'll overwrite this per chunk)
        DCDReporter(str(dcd_path), report_interval_steps),
        StateDataReporter(
            str(log_path),
            report_interval_steps,
            step=True,
            potentialEnergy=True,
            temperature=True,
            volume=True,
            separator="\t",
        ),
    ]

    for mult in multipliers:
        chunk_ps = base_ps * mult
        chunk_steps = int(chunk_ps / timestep_ps)

        # live progress for this chunk
        sim.reporters[0] = StateDataReporter(
            sys.stdout,
            report_interval_steps,
            step=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=chunk_steps,
            temperature=True,
            potentialEnergy=True,
            volume=True,
            separator="\t",
        )

        sim.step(chunk_steps)

        chk_path = OUT_DIR / f"{name}_{solvent_choice}_chk_{int(chunk_ps)}ps.pdb"
        _write_snapshot_pdb_wrapped(sim, chk_path)
        print(f"[checkpoint] {name} {solvent_choice} +{chunk_ps} ps -> {chk_path.name}")


def main():
    for job in TARGETS:
        name = job["name"]

        preeq_chk_path = OUT_DIR / f"{name}_{SOLVENT_CHOICE}_preeq.chk"
        eq_chk_path = OUT_DIR / f"{name}_{SOLVENT_CHOICE}_eq.chk"

        minimized_pdb_path = OUT_DIR / f"{name}_{SOLVENT_CHOICE}_minimized.pdb"
        eq_pdb_path = OUT_DIR / f"{name}_{SOLVENT_CHOICE}_eq.pdb"

        dcd_path = OUT_DIR / f"{name}_{SOLVENT_CHOICE}.dcd"
        log_path = OUT_DIR / f"{name}_{SOLVENT_CHOICE}.log"

        # -------------------------------------------------
        # 0. Build solvated system with restraints ON
        # -------------------------------------------------
        sim = _build_solvated_sim(
            job=job,
            solvent_kind=SOLVENT_CHOICE,
            box_nm=BOX_EDGE_NM,
        )

        # -------------------------------------------------
        # 1. PRE-EQ (restrained solute heavy atoms)
        # -------------------------------------------------
        if preeq_chk_path.exists():
            with open(preeq_chk_path, "rb") as fh:
                sim.context.loadCheckpoint(fh.read())
            print(f"[resume] {name}: loaded {preeq_chk_path.name}")
        else:
            # minimize (restraints active so solute won't distort)
            sim.minimizeEnergy()
            _write_snapshot_pdb_wrapped(sim, minimized_pdb_path)

            # reporter for PRE_EQ
            sim.reporters = [
                StateDataReporter(
                    os.sys.stdout,
                    REPORT_INT_STEPS,
                    step=True,
                    progress=True,
                    remainingTime=True,
                    speed=True,
                    totalSteps=PRE_EQ_STEPS,
                    temperature=True,
                    potentialEnergy=True,
                    volume=True,
                    separator="\t",
                )
            ]

            sim.step(PRE_EQ_STEPS)

            # checkpoint after PRE_EQ
            with open(preeq_chk_path, "wb") as fh:
                fh.write(sim.context.createCheckpoint())
            print(f"[checkpoint] {name}: wrote {preeq_chk_path.name}")

        # -------------------------------------------------
        # Turn OFF restraints before EQ
        # -------------------------------------------------
        sim.context.setParameter(
            "k_restraint",
            0.0 * unit.kilojoule_per_mole / (unit.nanometer ** 2),
        )

        # -------------------------------------------------
        # 2. EQ (unrestrained NPT settle)
        # -------------------------------------------------
        if eq_chk_path.exists():
            with open(eq_chk_path, "rb") as fh:
                sim.context.loadCheckpoint(fh.read())
            print(f"[resume] {name}: loaded {eq_chk_path.name}")
        else:
            sim.reporters = [
                StateDataReporter(
                    os.sys.stdout,
                    REPORT_INT_STEPS,
                    step=True,
                    progress=True,
                    remainingTime=True,
                    speed=True,
                    totalSteps=EQ_STEPS,
                    temperature=True,
                    potentialEnergy=True,
                    volume=True,
                    separator="\t",
                )
            ]

            sim.step(EQ_STEPS)

            _write_snapshot_pdb_wrapped(sim, eq_pdb_path)

            with open(eq_chk_path, "wb") as fh:
                fh.write(sim.context.createCheckpoint())
            print(f"[checkpoint] {name}: wrote {eq_chk_path.name}")
            print(f"[eq] {name}: wrote {eq_pdb_path.name}")

        # -------------------------------------------------
        # 3. Production
        # -------------------------------------------------
        sim.reporters = [
            StateDataReporter(
                os.sys.stdout,
                REPORT_INT_STEPS,
                step=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=1,  # dummy, replaced per chunk
                temperature=True,
                potentialEnergy=True,
                volume=True,
                separator="\t",
            ),
            DCDReporter(str(dcd_path), REPORT_INT_STEPS),
            StateDataReporter(
                str(log_path),
                REPORT_INT_STEPS,
                step=True,
                potentialEnergy=True,
                temperature=True,
                volume=True,
                separator="\t",
            ),
        ]

        run_production_in_chunks(
            sim=sim,
            name=name,
            solvent_choice=SOLVENT_CHOICE,
            base_ps=1.0,
            multipliers=[1.0, 10.0, 100.0, 1000.0],
            timestep_ps=TIMESTEP_PS,
            report_interval_steps=REPORT_INT_STEPS,
            dcd_path=dcd_path,
            log_path=log_path,
        )

        print(f"[done] {name}: {dcd_path.name}, {log_path.name}, {eq_pdb_path.name} in {OUT_DIR}")


if __name__ == "__main__":
    main()
