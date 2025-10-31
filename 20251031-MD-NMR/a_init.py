# /home/ra/repos/playground/20251031-MD-NMR/a_init.py

from pathlib import Path

from bugs import mkdir
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdmolfiles import PDBWriter

OUT_DIR = mkdir(Path(__file__).with_suffix(''))

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"

STRYCHNINE = (
    "c1ccc2c(c1)[C@]34CC[N@@]5[C@H]3C[C@@H]6[C@@H]7[C@@H]4N2C(=O)C[C@@H]7OCC=C6C5"
)

TARGETS = [
    {'name': 'aspirin', 'smiles': ASPIRIN, 'pdb': 'aspirin_init.pdb'},
    {'name': 'strychnine', 'smiles': STRYCHNINE, 'pdb': 'strychnine_init.pdb'},
]


def _embed_and_minimize(smiles: str, name: str, n_confs: int = 1):
    """
    Generate a single reasonable 3D conformer for a molecule from SMILES.
    - adds hydrogens
    - ETKDG embed
    - MMFF94 optimize
    - returns (mol_with_coords, conf_id)
    """
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f'Cannot parse SMILES for {name}')

    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 12345
    cid = AllChem.EmbedMolecule(mol, params)

    if cid < 0:
        raise RuntimeError(f'Conformer gen failed for {name}')

    # MMFF optimize
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant='MMFF94s')

    if mmff_props is None:
        raise RuntimeError(f'No MMFF params for {name}')

    AllChem.MMFFOptimizeMolecule(
        mol,
        mmffVariant='MMFF94s',  # you were already trying to use 'MMFF94s'
        maxIters=500,
        confId=cid
    )

    return (mol, cid)


def _write_pdb(mol, conf_id: int, out_path: str):
    """
    Write the given conformer to PDB with 3D coords.
    """
    with PDBWriter(out_path) as w:
        w.write(mol, confId=conf_id)


# main / script logic

def main():
    for t in TARGETS:
        (mol, cid) = _embed_and_minimize(t['smiles'], t['name'])
        _write_pdb(mol, cid, OUT_DIR / t['pdb'])
        print(f"Wrote {t['pdb']}")

    print("Done. These PDBs are the starting geometries you can now load into OpenMM.")


if __name__ == "__main__":
    main()
