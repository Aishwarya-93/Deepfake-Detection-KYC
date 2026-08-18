"""
test_dataset_splitter.py

Lightweight, dependency-free (stdlib unittest + pandas) tests for the
identity-disjoint dataset splitter.

Run with:
    python -m unittest tests.test_dataset_splitter -v
or:
    python tests/test_dataset_splitter.py
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.preprocessing.dataset_splitter import (  # noqa: E402
    SplitValidationError,
    UnionFind,
    assign_core_groups_to_splits,
    build_core_identity_groups,
    build_dfd_actor_groups,
    compute_identity_overlap_matrix,
    parse_identity,
    split_dataset_identity_disjoint,
    summarize_identity_split,
    validate_identity_disjoint_split,
)

METADATA_PATH = PROJECT_ROOT / "data" / "metadata" / "master_metadata.csv"


class TestParseIdentity(unittest.TestCase):

    def test_core_real(self):
        namespace, identities = parse_identity("original/000.mp4")
        self.assertEqual(namespace, "core")
        self.assertEqual(identities, ("000",))

    def test_core_fake(self):
        namespace, identities = parse_identity("Deepfakes/000_003.mp4")
        self.assertEqual(namespace, "core")
        self.assertEqual(identities, ("000", "003"))

    def test_core_fake_all_methods(self):
        for method in [
            "Deepfakes", "FaceSwap", "Face2Face",
            "FaceShifter", "NeuralTextures",
        ]:
            namespace, identities = parse_identity(f"{method}/123_456.mp4")
            self.assertEqual(namespace, "core")
            self.assertEqual(identities, ("123", "456"))

    def test_dfd(self):
        namespace, identities = parse_identity(
            "DeepFakeDetection/01_02__meeting_serious__YVGY8LOK.mp4"
        )
        self.assertEqual(namespace, "dfd")
        self.assertEqual(identities, ("01", "02"))

    def test_unrecognised_folder_raises(self):
        with self.assertRaises(ValueError):
            parse_identity("SomeOtherFolder/000_003.mp4")

    def test_malformed_filename_raises(self):
        with self.assertRaises(ValueError):
            parse_identity("original/not_a_valid_name.mp4")

    def test_missing_folder_raises(self):
        with self.assertRaises(ValueError):
            parse_identity("000.mp4")


class TestUnionFind(unittest.TestCase):

    def test_union_and_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        uf.add("d")

        components = uf.components()
        sizes = sorted(len(members) for members in components.values())
        self.assertEqual(sizes, [1, 3])

    def test_find_is_stable_after_union(self):
        uf = UnionFind()
        uf.union("x", "y")
        self.assertEqual(uf.find("x"), uf.find("y"))


class TestSyntheticGroupAssignment(unittest.TestCase):
    """Uses small synthetic metadata so this suite does not depend on
    the real 7,000-row master_metadata.csv being present."""

    def _synthetic_core_metadata(self, n_groups):
        rows = []
        methods = [
            "Deepfakes", "FaceSwap", "Face2Face",
            "FaceShifter", "NeuralTextures",
        ]
        for i in range(n_groups):
            id_a = f"{2 * i:03d}"
            id_b = f"{2 * i + 1:03d}"
            rows.append({"File Path": f"original/{id_a}.mp4", "Label": "REAL"})
            rows.append({"File Path": f"original/{id_b}.mp4", "Label": "REAL"})
            for method in methods:
                rows.append({
                    "File Path": f"{method}/{id_a}_{id_b}.mp4",
                    "Label": "FAKE",
                })
                rows.append({
                    "File Path": f"{method}/{id_b}_{id_a}.mp4",
                    "Label": "FAKE",
                })
        return pd.DataFrame(rows)

    def test_group_count_and_size(self):
        metadata = self._synthetic_core_metadata(10)
        groups = build_core_identity_groups(metadata)
        self.assertEqual(len(groups), 10)
        for members in groups.values():
            self.assertEqual(len(members), 2)

    def test_assignment_is_deterministic(self):
        metadata = self._synthetic_core_metadata(10)
        groups = build_core_identity_groups(metadata)

        assignment_1 = assign_core_groups_to_splits(
            groups, train_groups=6, validation_groups=2, test_groups=2,
            random_state=42,
        )
        assignment_2 = assign_core_groups_to_splits(
            groups, train_groups=6, validation_groups=2, test_groups=2,
            random_state=42,
        )
        self.assertEqual(assignment_1, assignment_2)

    def test_assignment_covers_every_group_exactly_once(self):
        metadata = self._synthetic_core_metadata(10)
        groups = build_core_identity_groups(metadata)

        assignment = assign_core_groups_to_splits(
            groups, train_groups=6, validation_groups=2, test_groups=2,
            random_state=42,
        )

        all_assigned = (
            assignment["train"] + assignment["validation"] + assignment["test"]
        )
        self.assertEqual(len(all_assigned), len(set(all_assigned)))
        self.assertEqual(set(all_assigned), set(groups.keys()))

    def test_wrong_group_total_raises(self):
        metadata = self._synthetic_core_metadata(10)
        groups = build_core_identity_groups(metadata)
        with self.assertRaises(ValueError):
            assign_core_groups_to_splits(
                groups, train_groups=6, validation_groups=2, test_groups=99,
                random_state=42,
            )


@unittest.skipUnless(
    METADATA_PATH.exists(),
    f"master_metadata.csv not found at {METADATA_PATH}",
)
class TestFullIdentityDisjointSplit(unittest.TestCase):
    """Integration test against the real master_metadata.csv."""

    @classmethod
    def setUpClass(cls):
        cls.metadata = pd.read_csv(METADATA_PATH)
        (
            cls.train_df,
            cls.validation_df,
            cls.test_df,
            cls.dfd_df,
            cls.report,
        ) = split_dataset_identity_disjoint(cls.metadata, random_state=42)

    def test_core_group_counts(self):
        core_groups = build_core_identity_groups(self.metadata)
        self.assertEqual(len(core_groups), 500)
        for members in core_groups.values():
            self.assertEqual(len(members), 2)

    def test_dfd_component_structure(self):
        dfd_groups = build_dfd_actor_groups(self.metadata)
        sizes = sorted(len(members) for members in dfd_groups.values())
        self.assertEqual(sizes, [5, 5, 18])

    def test_split_video_counts(self):
        self.assertEqual(len(self.train_df), 4200)
        self.assertEqual(len(self.validation_df), 900)
        self.assertEqual(len(self.test_df), 900)
        self.assertEqual(len(self.dfd_df), 1000)

    def test_split_label_counts(self):
        self.assertEqual(int((self.train_df["Label"] == "REAL").sum()), 700)
        self.assertEqual(int((self.train_df["Label"] == "FAKE").sum()), 3500)
        self.assertEqual(int((self.validation_df["Label"] == "REAL").sum()), 150)
        self.assertEqual(int((self.validation_df["Label"] == "FAKE").sum()), 750)
        self.assertEqual(int((self.test_df["Label"] == "REAL").sum()), 150)
        self.assertEqual(int((self.test_df["Label"] == "FAKE").sum()), 750)
        self.assertEqual(int((self.dfd_df["Label"] == "FAKE").sum()), 1000)

    def test_group_counts_per_split(self):
        self.assertEqual(self.train_df["Identity Group"].nunique(), 350)
        self.assertEqual(self.validation_df["Identity Group"].nunique(), 75)
        self.assertEqual(self.test_df["Identity Group"].nunique(), 75)

    def test_no_identity_overlap(self):
        matrix = compute_identity_overlap_matrix(
            self.train_df, self.validation_df, self.test_df
        )
        self.assertEqual(matrix.loc["train", "validation"], 0)
        self.assertEqual(matrix.loc["train", "test"], 0)
        self.assertEqual(matrix.loc["validation", "test"], 0)

    def test_validation_passes_on_correct_split(self):
        self.assertTrue(
            validate_identity_disjoint_split(
                self.train_df, self.validation_df, self.test_df, self.dfd_df
            )
        )

    def test_validation_fails_loudly_on_injected_leak(self):
        leaked_row = self.validation_df.iloc[[0]]
        tampered_train_df = pd.concat(
            [self.train_df, leaked_row], ignore_index=True
        )
        with self.assertRaises(SplitValidationError):
            validate_identity_disjoint_split(
                tampered_train_df, self.validation_df, self.test_df, self.dfd_df
            )

    def test_validation_fails_loudly_on_dfd_in_train(self):
        leaked_dfd_row = self.dfd_df.iloc[[0]]
        tampered_train_df = pd.concat(
            [self.train_df, leaked_dfd_row], ignore_index=True
        )
        with self.assertRaises(SplitValidationError):
            validate_identity_disjoint_split(
                tampered_train_df, self.validation_df, self.test_df, self.dfd_df
            )

    def test_validation_fails_loudly_on_missing_group(self):
        shrunk_train_df = self.train_df[
            self.train_df["Identity Group"] != self.train_df["Identity Group"].iloc[0]
        ]
        with self.assertRaises(SplitValidationError):
            validate_identity_disjoint_split(
                shrunk_train_df, self.validation_df, self.test_df, self.dfd_df
            )

    def test_summary_matches_expected_counts(self):
        summary = summarize_identity_split(
            self.train_df, self.validation_df, self.test_df, self.dfd_df
        )
        self.assertEqual(summary["videos_per_split"]["train"], 4200)
        self.assertEqual(summary["videos_per_split"]["validation"], 900)
        self.assertEqual(summary["videos_per_split"]["test"], 900)
        self.assertEqual(summary["dfd_video_count"], 1000)
        self.assertEqual(summary["unique_identities_per_split"]["train"], 700)
        self.assertEqual(summary["unique_identities_per_split"]["validation"], 150)
        self.assertEqual(summary["unique_identities_per_split"]["test"], 150)


if __name__ == "__main__":
    unittest.main(verbosity=2)
