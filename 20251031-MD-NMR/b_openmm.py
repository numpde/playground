# /home/ra/repos/playground/20251031-MD-NMR/b_openmm.py

from pathlib import Path

import numpy as np
import openmm
import openmm.unit as unit
from openmm.app import Simulation, DCDReporter, StateDataReporter

from openff.toolkit import Molecule, Topology, ForceField
from openff.units import unit as offunit

from rdkit import Chem
from rdkit.Chem import AllChem

from bugs import mkdir
from a_init import TARGETS  # pulls in [{'name','smiles','pdb'}, ...]

# Make an output dir for THIS script, analogous to a_init.OUT_DIR.
# For b_openmm.py this will be something like ".../b_openmm"
OUT_DIR = mkdir(Path(__file__).with_suffix(''))


def _rdkit_embed_minimize(smiles: str, name: str):
    """
    RDKit:
    - SMILES -> molecule with explicit Hs
    - ETKDGv3 embed
    - MMFF94s geometry refine
    Returns (rdmol, coords_Å) with coords as (n_atoms,3) float64 in Å.
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Bad SMILES for {name}")
    rdmol = Chem.AddHs(rdmol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 12345
    cid = AllChem.EmbedMolecule(rdmol, params)
    if cid < 0:
        raise RuntimeError(f"Conformer gen failed for {name}")

    # MMFF optimization (modern RDKit signature)
    AllChem.MMFFOptimizeMolecule(
        rdmol,
        mmffVariant='MMFF94s',
        maxIters=500,
        confId=cid,
    )

    conf = rdmol.GetConformer(cid)
    n_atoms = rdmol.GetNumAtoms()
    coords = np.zeros((n_atoms, 3), dtype=float)

    for i in range(n_atoms):
        p = conf.GetAtomPosition(i)  # in Å
        coords[i, 0] = p.x
        coords[i, 1] = p.y
        coords[i, 2] = p.z

    return (rdmol, coords)


def _build_openmm_sim(smiles: str, name: str, ff_name: str = "openff-2.0.0.offxml"):
    """
    1. RDKit -> 3D coords
    2. OpenFF Molecule (chemistry: bonds, charges, etc.)
    3. Attach coords as a conformer
    4. Topology.from_molecules(...)
    5. ForceField(ff_name).create_openmm_system(top)
    6. Wrap in OpenMM Simulation w/ Langevin thermostat at 300 K
    """

    # 1) RDKit geometry
    rdmol, coords_A = _rdkit_embed_minimize(smiles, name)

    # 2) OpenFF molecule (keep Hs explicit to preserve atom ordering)
    off_mol = Molecule.from_rdkit(rdmol, hydrogens_are_explicit=True)

    # 3) Attach the RDKit conformer (expects units)
    off_mol.add_conformer(coords_A * offunit.angstrom)

    # 4) Build topology
    top_off = Topology.from_molecules([off_mol])

    # 5) Parameterize with an OpenFF small-molecule FF
    ff = ForceField(ff_name)
    system = ff.create_openmm_system(top_off)  # -> openmm.System

    # 6) Prep OpenMM Simulation
    top_omm = top_off.to_openmm()

    # Langevin thermostat @300 K, 1/ps friction, 4 fs step
    integrator = openmm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.004 * unit.picoseconds,
    )

    sim = Simulation(top_omm, system, integrator)

    # Set starting coordinates
    # coords_A (Å) --> nm, because OpenMM uses nanometers
    coords_nm = (coords_A * 0.1) * unit.nanometer
    sim.context.setPositions(coords_nm)

    return sim


def main():
    # TARGETS came from a_init.py:
    # [
    #   {'name': 'aspirin', 'smiles': <...>, 'pdb': 'aspirin_init.pdb'},
    #   {'name': 'strychnine', 'smiles': <...>, 'pdb': 'strychnine_init.pdb'},
    # ]

    for job in TARGETS:
        name = job["name"]
        smiles = job["smiles"]

        sim = _build_openmm_sim(smiles, name)

        # quick vacuum minimization
        sim.minimizeEnergy()

        # attach reporters: short trajectory + thermodynamics
        dcd_path = OUT_DIR / f"{name}_vacuum.dcd"
        log_path = OUT_DIR / f"{name}_vacuum.log"

        sim.reporters.append(DCDReporter(str(dcd_path), 100))
        sim.reporters.append(
            StateDataReporter(
                str(log_path),
                100,
                step=True,
                potentialEnergy=True,
                temperature=True,
            )
        )

        # ~40 ps MD test (10000 steps * 4 fs/step)
        sim.step(10_000)

        print(f"{name}: wrote {dcd_path.name} and {log_path.name} in {OUT_DIR}")

    print("Done. You now have short MD trajectories for both molecules in (gas-phase) MM.")


if __name__ == "__main__":
    main()
