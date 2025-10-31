# /home/ra/repos/playground/20251031-MD-NMR/b_openmm.py

from pathlib import Path
from typing import Tuple

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

import a_init
from a_init import TARGETS  # [{'name','smiles','pdb','charge',...}, ...]

# a_init writes its PDBs into a_init/ (same stem as a_init.py)
A_INIT_DIR = Path(a_init.__file__).with_suffix('')

# b_openmm output dir (b_openmm/)
OUT_DIR = mkdir(Path(__file__).with_suffix(''))

# force field to parameterize small organics
OPENFF_FF = "openff-2.0.0.offxml"

# Langevin thermostat settings
TARGET_TEMP = 300.0 * unit.kelvin
FRICTION = 1.0 / unit.picosecond
DT = 0.004 * unit.picoseconds  # 4 fs (safe enough for our short sanity runs)
STEPS = 10_000  # ~40 ps at 4 fs/step
REPORT_INTERVAL = 100  # write every 100 steps


def _load_rdkit_from_smiles(smiles: str) -> Chem.Mol:
    """
    Rebuild RDKit mol from SMILES (explicit Hs, same atom order as a_init).
    We are assuming the SMILES we get here is the same one a_init used
    to generate the *_init.pdb, so atom ordering matches.
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Bad SMILES: {smiles}")
    rdmol = Chem.AddHs(rdmol)
    return rdmol


def _load_coords_from_pdb(pdb_path: Path) -> Tuple[np.ndarray, openmm.app.Topology]:
    """
    Read the PDB we wrote in a_init for this protomer.
    Returns:
      coords_A : (N,3) float64 in Å
      top_pdb  : OpenMM Topology from that file
    Note: RDKit's PDBWriter wrote coords in Å. PDBFile reports nm.
    """
    pdb = PDBFile(str(pdb_path))
    # positions as Quantity in nm
    pos_nm = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)  # (N,3)
    coords_A = pos_nm * 10.0  # 1 nm = 10 Å
    return (coords_A, pdb.topology)


def _make_openff_mol_with_conformer(rdmol: Chem.Mol, coords_A: np.ndarray) -> Molecule:
    """
    Convert RDKit mol -> OpenFF Molecule, attach conformer (Å).
    """
    off_mol = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)
    off_mol.add_conformer(coords_A * offunit.angstrom)
    return off_mol


def _build_openmm_sim(off_mol: Molecule) -> Tuple[Simulation, openmm.System]:
    """
    Given an OpenFF Molecule (with conformer), build:
      - OpenFF Topology
      - parameterized openmm.System
      - Simulation with Langevin thermostat
    Set initial positions in nm.
    """
    top_off = Topology.from_molecules([off_mol])

    ff = ForceField(OPENFF_FF)
    system = ff.create_openmm_system(top_off)

    top_omm = top_off.to_openmm()

    # Rename residue to 'LIG' so downstream (c_solvated.py) can find/lock the solute
    for chain in top_omm.chains():
        for res in chain.residues():
            res.name = "LIG"

    integrator = openmm.LangevinMiddleIntegrator(
        TARGET_TEMP,
        FRICTION,
        DT,
    )

    sim = Simulation(top_omm, system, integrator)

    # Use the first (and only) conformer from off_mol for coordinates
    coords_A = off_mol.conformers[0].m_as(offunit.angstrom)  # Å -> plain float
    coords_nm = (coords_A * 0.1) * unit.nanometer  # Å *0.1 = nm
    sim.context.setPositions(coords_nm)

    return (sim, system)


def _run_short_vacuum_md(name: str, sim: Simulation) -> None:
    """
    Minimize, then short Langevin MD in vacuum.
    Write DCD + log into b_openmm/.
    """
    # relax in-place
    sim.minimizeEnergy()

    dcd_path = OUT_DIR / f"{name}_vacuum.dcd"
    log_path = OUT_DIR / f"{name}_vacuum.log"

    sim.reporters.append(DCDReporter(str(dcd_path), REPORT_INTERVAL))
    sim.reporters.append(
        StateDataReporter(
            str(log_path),
            REPORT_INTERVAL,
            step=True,
            potentialEnergy=True,
            temperature=True,
        )
    )

    sim.step(STEPS)

    print(f"{name}: wrote {dcd_path.name}, {log_path.name} in {OUT_DIR}")


def main():
    """
    For each protomer/tautomer we care about (from a_init.TARGETS):
      - load its a_init/<name>_init.pdb
      - rebuild the OpenFF molecule with that geometry
      - build an OpenMM Simulation in vacuum
      - quick sanity MD run
    This is mainly a smoke test: are parameters sane, does it integrate stably?
    """

    for job in TARGETS:
        name = job["name"]  # e.g. 'aspirin_neutral'
        smiles = job["smiles"]
        pdb_rel = job["pdb"]  # e.g. 'aspirin_neutral_init.pdb'
        pdb_path = A_INIT_DIR / pdb_rel

        # coords from the PDB we already wrote in a_init
        (coords_A, _top_from_pdb) = _load_coords_from_pdb(pdb_path)

        # rebuild chemistry from SMILES, attach those coords
        rdmol = _load_rdkit_from_smiles(smiles)
        off_mol = _make_openff_mol_with_conformer(rdmol, coords_A)

        # OpenMM Simulation for this single solute in vacuum
        (sim, _system) = _build_openmm_sim(off_mol)

        # sanity run
        _run_short_vacuum_md(name, sim)

    print("Done. Vacuum sanity MD complete for all TARGETS.")


if __name__ == "__main__":
    main()
