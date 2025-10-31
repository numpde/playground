# /home/ra/repos/playground/20251031-MD-NMR/c_solvated.py

"""
c_solvated.py

Goal:
- Build aspirin / strychnine in explicit solvent (water / DMSO / CDCl3)
- Pack solvent at ~liquid density
- Run NPT (300 K, 1 bar) using OpenMM on multithreaded CPU
- Dump:
    *_minimized.pdb   (after energy minimization)
    *_eq.pdb          (after equilibration, use this as PyMOL reference)
    *.dcd / *.log     (production trajectory + thermodynamics)

Stability changes vs last version:
- pack solvent on a grid at ~correct density (no huge vacuum / no barostat collapse)
- 1 fs timestep (safer; 4 fs was too aggressive for generic organics)
- two-phase equilibration (short settle + longer NPT)
- explicit CPU platform with all cores
"""

import math
import os
import sys
from pathlib import Path

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

from a_init import TARGETS  # [{'name','smiles','pdb'}, ...]

# output dir: ./c_solvated  (basename of this file)
OUT_DIR = mkdir(Path(__file__).with_suffix(''))


# ---------- geometry / conversion helpers ----------

def _rdkit_embed_minimize(smiles: str, label: str):
    """
    RDKit:
    - SMILES -> mol with explicit H
    - embed 3D (ETKDGv3)
    - MMFF94s geometry refine
    returns (rdmol, coords_A) with coords in Å
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Bad SMILES for {label}")
    rdmol = Chem.AddHs(rdmol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 12345
    cid = AllChem.EmbedMolecule(rdmol, params)
    if cid < 0:
        raise RuntimeError(f"Conformer gen failed for {label}")

    AllChem.MMFFOptimizeMolecule(
        rdmol,
        mmffVariant="MMFF94s",
        maxIters=500,
        confId=cid,
    )

    conf = rdmol.GetConformer(cid)
    n_atoms = rdmol.GetNumAtoms()
    coords_A = np.zeros((n_atoms, 3), dtype=float)
    for i in range(n_atoms):
        p = conf.GetAtomPosition(i)  # Å
        coords_A[i] = (p.x, p.y, p.z)

    return (rdmol, coords_A)


def _off_mol_from_rdkit_with_coords(rdmol, coords_A):
    """
    RDKit mol -> OpenFF Molecule with same atom ordering + conformer.
    coords_A: (N,3) Å
    """
    off_mol = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)
    off_mol.add_conformer(coords_A * offunit.angstrom)
    return off_mol


def _center_coords(coords_nm: np.ndarray) -> np.ndarray:
    """
    Translate coords so centroid ~0.
    coords_nm: (N,3) in nm
    """
    ctr = coords_nm.mean(axis=0, keepdims=True)
    return coords_nm - ctr


def _make_box_vectors(box_nm: float):
    """
    Build a cubic periodic box of edge length box_nm [nm].
    Returns (3,3) np.array.
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


# ---------- solvent packing helpers ----------

def _estimate_n_solvent(solvent_kind: str, box_nm: float) -> int:
    """
    Approximate liquid-like number of solvent molecules in a cubic box_nm^3 nm^3.

    Bulk molar densities (mol/L):
      water:   ~55.5
      dmso:    ~14.0   (ρ≈1.10 g/mL, M≈78)
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


def _prepare_solute(solute_smiles: str, solute_name: str):
    """
    Build OpenFF Molecule for solute + centered coords in nm.
    """
    (solute_rdmol, solute_coords_A) = _rdkit_embed_minimize(solute_smiles, solute_name)
    off_solute = _off_mol_from_rdkit_with_coords(solute_rdmol, solute_coords_A)

    solute_coords_nm = solute_coords_A * 0.1  # Å -> nm
    solute_coords_nm = _center_coords(solute_coords_nm)

    return (off_solute, solute_coords_nm)


def _prepare_solvent_template(solvent_kind: str):
    """
    Return (off_solvent, template_coords_nm_centered)
    """
    if solvent_kind == "water":
        solvent_smiles = "O"  # H2O
    elif solvent_kind == "dmso":
        solvent_smiles = "CS(=O)C"  # DMSO
    elif solvent_kind == "cdcl3":
        solvent_smiles = "ClC(Cl)Cl"  # CHCl3 ~= CDCl3 (ignore isotope)
    else:
        raise ValueError(f"Unknown solvent_kind {solvent_kind}")

    (solv_rdmol, solv_coords_A) = _rdkit_embed_minimize(solvent_smiles, solvent_kind)
    off_solvent = _off_mol_from_rdkit_with_coords(solv_rdmol, solv_coords_A)

    solv_coords_nm = solv_coords_A * 0.1  # Å -> nm
    solv_coords_nm = _center_coords(solv_coords_nm)

    return (off_solvent, solv_coords_nm)


def _place_solvent_grid(
        template_coords_nm: np.ndarray,
        n_copies: int,
        box_nm: float,
        min_dist_from_origin_nm: float = 0.4,
):
    """
    Place n_copies solvent molecules in a simple 3D lattice.

    - tile box into n_per_dim^3 cells
    - place the solvent COM at each cell center
    - skip cells too close to solute center (< min_dist_from_origin_nm)
    - if we still need more, fill remaining cells (fallback)

    Returns:
      coords_list_nm: list of (Ns,3) arrays for each solvent copy in nm
    """
    Ns = template_coords_nm.shape[0]

    n_per_dim = math.ceil(n_copies ** (1.0 / 3.0))
    capacity = n_per_dim ** 3
    if capacity < n_copies:
        n_per_dim += 1
        capacity = n_per_dim ** 3

    cell = box_nm / n_per_dim  # nm per grid cell

    centers = []
    for ix in range(n_per_dim):
        for iy in range(n_per_dim):
            for iz in range(n_per_dim):
                cx = (ix + 0.5) / n_per_dim - 0.5
                cy = (iy + 0.5) / n_per_dim - 0.5
                cz = (iz + 0.5) / n_per_dim - 0.5
                centers.append((cx * box_nm, cy * box_nm, cz * box_nm))

    rng = np.random.default_rng(12345)
    rng.shuffle(centers)

    coords_list_nm = []
    used = 0
    for (cx, cy, cz) in centers:
        if used >= n_copies:
            break
        dist0 = math.sqrt(cx * cx + cy * cy + cz * cz)
        if dist0 < min_dist_from_origin_nm:
            continue
        shift = np.array([[cx, cy, cz]], dtype=float)  # (1,3)
        this_coords = template_coords_nm + shift  # (Ns,3)
        coords_list_nm.append(this_coords)
        used += 1

    if used < n_copies:
        # fallback: fill remaining cells even if close to origin
        for (cx, cy, cz) in centers:
            if used >= n_copies:
                break
            shift = np.array([[cx, cy, cz]], dtype=float)
            this_coords = template_coords_nm + shift
            coords_list_nm.append(this_coords)
            used += 1

    return coords_list_nm


# ---------- OpenMM system builder ----------

def _build_solvated_sim(
        solute_smiles: str,
        solute_name: str,
        solvent_kind: str = "water",  # "water", "dmso", "cdcl3"
        box_nm: float = 3.0,
):
    """
    Create an OpenMM Simulation (CPU platform, multithreaded) of `solute_name`
    solvated in `solvent_kind` at 300 K / 1 bar.

    Steps:
    - build solute coords (centered)
    - build solvent template coords (centered)
    - estimate number of solvent molecules for ~liquid density
    - pack solvent on a lattice
    - create OpenFF Topology w/ periodic box
    - create OpenMM System via OpenFF ForceField
    - add barostat (1 bar / 300 K)
    - LangevinMiddleIntegrator @ 300 K
    - CPU platform with all cores
    - set initial coords and periodic box vectors
    """

    # 1) solute
    (off_solute, solute_coords_nm) = _prepare_solute(solute_smiles, solute_name)

    # 2) solvent template + count
    (off_solvent, solvent_template_nm) = _prepare_solvent_template(solvent_kind)
    n_solvent = _estimate_n_solvent(solvent_kind, box_nm)

    # 3) place solvent
    solvent_coords_list_nm = _place_solvent_grid(
        solvent_template_nm,
        n_copies=n_solvent,
        box_nm=box_nm,
    )

    # 4) stitch coords: [solute] + [solvent_i ...]
    molecules_for_top = [off_solute] + [off_solvent] * len(solvent_coords_list_nm)

    coords_all_nm = np.concatenate(
        [solute_coords_nm] + solvent_coords_list_nm,
        axis=0,
    )  # shape (Ntotal,3) in nm

    # 5) topology w/ PBC
    box_vectors = _make_box_vectors(box_nm)  # (3,3)
    top_off = Topology.from_molecules(molecules_for_top)
    top_off.box_vectors = box_vectors * offunit.nanometer

    # 6) force field
    # TIP3P is needed for water; other solvents come from general FF
    if solvent_kind == "water":
        ff = ForceField("openff-2.0.0.offxml", "tip3p.offxml")
    else:
        ff = ForceField("openff-2.0.0.offxml")

    system = ff.create_openmm_system(top_off)

    # barostat for NPT (1 bar / 300 K)
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin))

    # 7) integrator
    # Stable timestep: 1 fs (0.001 ps)
    # friction 1/ps is fine for Langevin thermostat
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picoseconds,  # 1 fs
    )

    # 8) multithreaded CPU platform
    platform = openmm.Platform.getPlatformByName("CPU")
    n_threads = os.cpu_count() or 1
    platform_props = {"Threads": str(n_threads)}

    # 9) Simulation
    top_omm = top_off.to_openmm()
    sim = Simulation(top_omm, system, integrator, platform, platform_props)

    # 10) initial coordinates + periodic box
    sim.context.setPositions(coords_all_nm * unit.nanometer)
    a_vec = box_vectors[0] * unit.nanometer
    b_vec = box_vectors[1] * unit.nanometer
    c_vec = box_vectors[2] * unit.nanometer
    sim.context.setPeriodicBoxVectors(a_vec, b_vec, c_vec)

    return sim


def _write_snapshot_pdb_wrapped(sim, out_path):
    """
    Periodic imaging for nice visualization:
    - keep every residue (water, solute) whole
    - place each residue's center into the primary unit cell [0,L)
    Assumes an orthorhombic box (diagonal box vectors).
    """
    # grab positions and box
    state = sim.context.getState(getPositions=True)
    pos_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # (N,3)
    (avec, bvec, cvec) = sim.context.getState().getPeriodicBoxVectors(asNumpy=True)

    box_lengths = np.array([
        avec[0].value_in_unit(unit.nanometer),
        bvec[1].value_in_unit(unit.nanometer),
        cvec[2].value_in_unit(unit.nanometer),
    ])  # [Lx, Ly, Lz] nm

    # we'll build a new coord array we can edit
    wrapped_nm = np.zeros_like(pos_nm)

    # iterate residues (each water is its own residue; solute should be one residue)
    top = sim.topology
    for res in top.residues():
        # collect atom indices for this residue
        idxs = [atom.index for atom in res.atoms()]
        coords = pos_nm[idxs, :]  # (n_res_atoms,3)

        # compute residue "center"
        center = coords.mean(axis=0)  # nm

        # wrap center into primary cell
        center_wrapped = center - np.floor(center / box_lengths) * box_lengths

        # shift all atoms by the same delta
        shift = center_wrapped - center
        wrapped_nm[idxs, :] = coords + shift

    # convert back to OpenMM Quantity and write PDB
    wrapped_q = wrapped_nm * unit.nanometer
    with open(out_path, "w") as fh:
        PDBFile.writeFile(top, wrapped_q, fh, keepIds=True)


def run_production_in_chunks(
        sim,
        name: str,
        solvent_choice: str,
        base_ps: float = 1.0,
        multipliers: list[float] = [1.0, 10.0, 100.0, 1000.0],
        timestep_ps: float = 0.001,
        report_interval_steps: int = 1000,
        dcd_path: Path | None = None,
        log_path: Path | None = None,
):
    """
    Run production MD in geometric time chunks.
    After each chunk, write a snapshot PDB so we can inspect convergence.

    sim                : OpenMM Simulation (already equilibrated, barostat/thermostat on)
    name, solvent_choice : tags for filenames
    base_ps            : smallest chunk length in picoseconds
    multipliers        : geometric schedule in units of base_ps
    timestep_ps        : integration step in ps (1 fs = 0.001 ps)
    report_interval_steps : how often to write to reporters
    dcd_path / log_path   : files for continuous trajectory + thermodynamics
    """

    import sys

    # attach reporters ONCE so DCD/log is continuous across chunks
    sim.reporters = [
        # live progress to stdout for current chunk (we'll replace this per chunk)
        # placeholder; we'll overwrite this element in the loop
        None,
        # trajectory
        DCDReporter(str(dcd_path), report_interval_steps),
        # logfile
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

    # index 0 in sim.reporters will be swapped each chunk with a chunk-scoped progress reporter

    for mult in multipliers:
        chunk_ps = base_ps * mult
        chunk_steps = int(chunk_ps / timestep_ps)

        # replace sim.reporters[0] with a fresh StateDataReporter
        sim.reporters[0] = StateDataReporter(
            sys.stdout,
            report_interval_steps,
            step=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=chunk_steps,  # so this chunk prints 0→100%
            temperature=True,
            potentialEnergy=True,
            volume=True,
            separator="\t",
        )

        # integrate this chunk
        sim.step(chunk_steps)

        # checkpoint snapshot after this chunk
        chk_path = OUT_DIR / f"{name}_{solvent_choice}_chk_{int(chunk_ps)}ps.pdb"
        _write_snapshot_pdb_wrapped(sim, chk_path)
        print(f"[checkpoint] {name} {solvent_choice} @ +{chunk_ps} ps -> {chk_path.name}")


# ---------- main ----------

def main():
    solvent_choice = "water"  # "water", "dmso", "cdcl3"
    box_nm = 3.0  # nm cube edge

    # timestep = 1 fs = 0.001 ps
    TIMESTEP_PS = 0.001

    PRE_EQ_PS = 50.0  # short pre-equilibration
    EQ_PS = 950.0  # long NPT settle
    REPORT_INT_STEPS = 1000  # report every 1000 steps (~1 ps at 1 fs)

    PRE_EQ_STEPS = int(PRE_EQ_PS / TIMESTEP_PS)  # 50,000
    EQ_STEPS = int(EQ_PS / TIMESTEP_PS)  # 950,000

    for job in TARGETS:
        name = job["name"]
        smiles = job["smiles"]

        # File paths for checkpoints / outputs
        preeq_chk_path = OUT_DIR / f"{name}_{solvent_choice}_preeq.chk"
        eq_chk_path = OUT_DIR / f"{name}_{solvent_choice}_eq.chk"

        minimized_pdb_path = OUT_DIR / f"{name}_{solvent_choice}_minimized.pdb"
        eq_pdb_path = OUT_DIR / f"{name}_{solvent_choice}_eq.pdb"

        dcd_path = OUT_DIR / f"{name}_{solvent_choice}.dcd"
        log_path = OUT_DIR / f"{name}_{solvent_choice}.log"

        # ---------------------------
        # 0. Build solvated system
        # ---------------------------
        sim = _build_solvated_sim(
            solute_smiles=smiles,
            solute_name=name,
            solvent_kind=solvent_choice,
            box_nm=box_nm,
        )

        # ---------------------------
        # 1. PRE-EQ phase
        # ---------------------------
        if os.path.exists(preeq_chk_path):
            # resume from PRE_EQ checkpoint
            with open(preeq_chk_path, "rb") as fh:
                sim.context.loadCheckpoint(fh.read())
            print(f"[resume] {name}: loaded {preeq_chk_path.name}")
        else:
            # fresh run: minimize then pre-equilibrate
            sim.minimizeEnergy()
            _write_snapshot_pdb_wrapped(sim, minimized_pdb_path)

            # attach reporter for PRE_EQ
            sim.reporters = [
                StateDataReporter(
                    sys.stdout,
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

            # save checkpoint after PRE_EQ
            with open(preeq_chk_path, "wb") as fh:
                fh.write(sim.context.createCheckpoint())
            print(f"[checkpoint] {name}: wrote {preeq_chk_path.name}")

        # ---------------------------
        # 2. EQ phase (long NPT settle)
        # ---------------------------
        if os.path.exists(eq_chk_path):
            # resume from EQ checkpoint
            with open(eq_chk_path, "rb") as fh:
                sim.context.loadCheckpoint(fh.read())
            print(f"[resume] {name}: loaded {eq_chk_path.name}")
        else:
            # continue from end of PRE_EQ into EQ
            sim.reporters = [
                StateDataReporter(
                    sys.stdout,
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

            # snapshot at end of equilibration
            _write_snapshot_pdb_wrapped(sim, eq_pdb_path)

            # save checkpoint after EQ
            with open(eq_chk_path, "wb") as fh:
                fh.write(sim.context.createCheckpoint())
            print(f"[checkpoint] {name}: wrote {eq_chk_path.name}")
            print(f"[eq] {name}: wrote {eq_pdb_path.name}")

        # ---------------------------
        # 3. Production run (continuous trajectory)
        # ---------------------------
        # reporters for production:
        sim.reporters = [
            # live progress to stdout during production chunks
            # (we'll overwrite this entry inside run_production_in_chunks)
            # placeholder; run_production_in_chunks will mutate sim.reporters[0]
            StateDataReporter(
                sys.stdout,
                REPORT_INT_STEPS,
                step=True,
                progress=True,
                remainingTime=True,
                speed=True,
                totalSteps=1,  # dummy; will get replaced per chunk
                temperature=True,
                potentialEnergy=True,
                volume=True,
                separator="\t",
            ),
            # trajectory file (keeps appending frames)
            DCDReporter(str(dcd_path), REPORT_INT_STEPS),
            # thermo log file
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

        # Now run geometric production chunks (1 ps, 10 ps, 100 ps, 1000 ps, ...)
        run_production_in_chunks(
            sim=sim,
            name=name,
            solvent_choice=solvent_choice,
            base_ps=1.0,
            multipliers=[1.0, 10.0, 100.0, 1000.0],
            timestep_ps=TIMESTEP_PS,
            report_interval_steps=REPORT_INT_STEPS,
            dcd_path=dcd_path,
            log_path=log_path,
        )

        print(f"[done] {name}: {dcd_path.name}, {log_path.name}, "
              f"{eq_pdb_path.name} in {OUT_DIR}")


if __name__ == "__main__":
    main()
