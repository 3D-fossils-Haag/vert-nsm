"""Tests for vtk_parsing_logic.py (filename -> family / genus / species).

Used by create_latents_per_group.py and NSM/train/train_deep_sdf_contrastive.py, so a
parsing change silently changes which taxon every vertebra is grouped under.
"""
import os
import pytest
from vtk_parsing_logic import parse_vtk_filename

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAPPING_CSV = os.path.join(REPO_ROOT, "vtk_name_to_mapping_v2.csv")
EMPTY = {"family": None, "genus": None, "species": None}


@pytest.mark.parametrize("name, expected", [
    ("agamidae_agama_atra_uf180711_06-t1.vtk", ("agamidae", "agama", "atra")),
    ("agamidae_agama_atra_uf180711_01-c3.vtk", ("agamidae", "agama", "atra")),
    ("varanidae_varanus_komodoensis_ufh2_03-l2.vtk", ("varanidae", "varanus", "komodoensis")),
    ("iguanidae_iguana_iguana_uf1_12-C3.vtk", ("iguanidae", "iguana", "iguana")),  # case-insensitive
])
def test_parses_standard_filenames(name, expected):
    assert parse_vtk_filename(name) == dict(zip(("family", "genus", "species"), expected))


def test_ignores_directories():
    path = os.path.join("some", "dir_with_underscores", "agamidae_agama_atra_uf180711_06-t1.vtk")
    assert parse_vtk_filename(path)["genus"] == "agama"


@pytest.mark.parametrize("name", ["readme.txt", "x.vtk", "", "no-vertebra-here.vtk"])
def test_unparseable_names_return_none(name):
    assert parse_vtk_filename(name) == EMPTY


@pytest.mark.xfail(strict=True, reason="KNOWN LIMITATION: with no species field the specimen ID "
                                       "is returned as the species. Decide whether that should be "
                                       "None, then remove this marker.")
def test_missing_species_is_not_filled_with_specimen_id():
    assert parse_vtk_filename("agamidae_agama_uf1_01-c3.vtk")["species"] is None


def test_matches_every_row_of_mapping_csv():
    """Regression test against the real dataset: the parser must agree with the curated
    family/genus/species columns for every mesh in vtk_name_to_mapping_v2.csv."""
    pd = pytest.importorskip("pandas")
    if not os.path.isfile(MAPPING_CSV):
        pytest.skip("vtk_name_to_mapping_v2.csv not present (it is matched by *.csv in .gitignore)")
    table = pd.read_csv(MAPPING_CSV)
    mismatches = [
        (row.vtk_name, parse_vtk_filename(row.vtk_name))
        for row in table.itertuples()
        if parse_vtk_filename(row.vtk_name) != {"family": row.family, "genus": row.genus, "species": row.species}
    ]
    assert len(table) > 0
    assert not mismatches, f"{len(mismatches)} of {len(table)} filenames parse differently, e.g. {mismatches[:5]}"
