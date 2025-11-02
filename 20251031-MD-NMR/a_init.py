# /home/ra/repos/playground/20251031-MD-NMR/a_init.py

from pathlib import Path
from typing import List, Dict, Tuple

from bugs import mkdir
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import PDBWriter

try:
    import gpu4pyscf
except ImportError:
    print("Warning: module gpu4pyscf not found")

# ---------------------------------------------------------------------
# config / output
# ---------------------------------------------------------------------

OUT_DIR = mkdir(Path(__file__).with_suffix(''))

# Tiny fast sanity-check system
ETHANOL_NEUTRAL = "CCO"  # CH3-CH2-OH, neutral

# Aspirin:
# - neutral (COOH)
# - deprotonated (COO-)  ← dominant around pH ~7 in water
ASPIRIN_NEUTRAL = "CC(=O)OC1=CC=CC=C1C(=O)O"
ASPIRIN_DEPROT = None  # "CC(=O)OC1=CC=CC=C1C(=O)[O-]"

# Strychnine:
# You currently use this neutral SMILES (free base). :contentReference[oaicite:1]{index=1}
STRYCHNINE_NEUTRAL = (
    "c1ccc2c(c1)[C@]34CC[N@@]5[C@H]3C[C@@H]6[C@@H]7[C@@H]4N2C(=O)C[C@@H]7OCC=C6C5"
)

# Protonated (+1) strychnine (aqueous, ~physiological pH) is basically the same
# scaffold with one tertiary N → [NH+]. Getting the exact stereochem + which N
# gets protonated is chemistry-dependent; put your best annotated SMILES here.
# TODO: fill this with the correct [NH+] form you want to simulate in water.
STRYCHNINE_PROTONATED = None  # placeholder


def _embed_and_minimize(smiles: str, name: str) -> Tuple[Chem.Mol, int]:
    """
    SMILES -> 3D RDKit mol:
    - add H
    - ETKDGv3 embed (seeded)
    - MMFF94s optimize
    returns (mol_with_coords, conf_id)
    """
    if smiles is None:
        raise ValueError(f"{name}: no SMILES provided")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'Cannot parse SMILES for {name}')

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 12345
    cid = AllChem.EmbedMolecule(mol, params)
    if cid < 0:
        raise RuntimeError(f'Conformer gen failed for {name}')

    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')
    if mmff_props is None:
        raise RuntimeError(f'No MMFF params for {name}')

    AllChem.MMFFOptimizeMolecule(
        mol,
        mmffVariant='MMFF94s',
        maxIters=500,
        confId=cid
    )

    return (mol, cid)


def _write_pdb(mol: Chem.Mol, conf_id: int, out_path: Path) -> None:
    """
    Dump the given conformer to PDB with 3D coords.
    """
    with PDBWriter(str(out_path)) as w:
        w.write(mol, confId=conf_id)


def _state_dict(name: str, smiles: str, charge_hint: int, pdb_name: str) -> Dict:
    """
    Convenience: one row for TARGETS.
    charge_hint is for downstream logic (e.g. water vs CDCl3 choice).
    """
    return {
        'name': name,
        'smiles': smiles,
        'pdb': pdb_name,
        'charge': charge_hint,
    }


def build_targets() -> List[Dict]:
    """
    Enumerate chemically relevant protomers / tautomers we care about.
    Each entry becomes its own system later (solvation, MD, clustering).
    """
    targets: List[Dict] = []

    # --- tiny / fast test case ---------------------------------
    targets.append(
        _state_dict(
            name='ethanol_neutral',
            smiles=ETHANOL_NEUTRAL,
            charge_hint=0,
            pdb_name='ethanol_neutral_init.pdb',
        )
    )

    # Aspirin
    targets.append(
        _state_dict(
            name='aspirin_neutral',  # nonpolar solvent, low pH
            smiles=ASPIRIN_NEUTRAL,
            charge_hint=0,
            pdb_name='aspirin_neutral_init.pdb',
        )
    )

    if ASPIRIN_DEPROT is not None:
        targets.append(
            _state_dict(
                name='aspirin_deprot',  # water-like / physiological pH
                smiles=ASPIRIN_DEPROT,
                charge_hint=-1,
                pdb_name='aspirin_deprot_init.pdb',
            )
        )

    # Strychnine
    targets.append(
        _state_dict(
            name='strychnine_neutral',  # free base, organic solvent
            smiles=STRYCHNINE_NEUTRAL,
            charge_hint=0,
            pdb_name='strychnine_neutral_init.pdb',
        )
    )

    if STRYCHNINE_PROTONATED is not None:
        targets.append(
            _state_dict(
                name='strychnine_protonated',  # aqueous, ~pH 7
                smiles=STRYCHNINE_PROTONATED,
                charge_hint=+1,
                pdb_name='strychnine_protonated_init.pdb',
            )
        )

    return targets


# This is what downstream scripts import.
TARGETS = build_targets()


def main():
    """
    For each target protonation state:
    - generate a single reasonable 3D conformer
    - write it to a_init/<name>_init.pdb
    """
    for t in TARGETS:
        (mol, cid) = _embed_and_minimize(t['smiles'], t['name'])
        out_path = OUT_DIR / t['pdb']
        _write_pdb(mol, cid, out_path)
        print(f"Wrote {out_path.name}  (charge~{t['charge']}, {t['name']})")

    print("Done. These PDBs are the starting geometries for OpenMM.")


if __name__ == "__main__":
    main()
