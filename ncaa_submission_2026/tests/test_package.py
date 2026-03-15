"""
test_package.py
===============
Integration and package completeness tests for ncaa_submission_2026.

Covers:
- All required files present
- run_prediction.py runs in synthetic mode
- Privacy certification generates JSON + MD with required sections
- Submission CSV passes Kaggle schema validation
- DependencyAuditor finds all required packages
- manifest.py: build_manifest, sha256_file, verify_manifest
- Full certification run passes all four checks on synthetic data
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
PKG = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Package file completeness
# ---------------------------------------------------------------------------

class TestPackageFiles:
    REQUIRED_FILES = [
        "README.md",
        "requirements.txt",
        "setup.py",
        "data_manifest.json",
        "model_manifest.json",
        "run_prediction.py",
        "__init__.py",
        "certification/__init__.py",
        "certification/leakage_detector.py",
        "certification/determinism_checker.py",
        "certification/network_isolator.py",
        "certification/dependency_auditor.py",
        "certification/certifier.py",
    ]

    def test_all_required_files_present(self):
        for relpath in self.REQUIRED_FILES:
            p = PKG / relpath
            assert p.exists(), f"Required file missing: {relpath}"

    def test_readme_has_quick_start(self):
        readme = (PKG / "README.md").read_text()
        assert "Quick Start" in readme or "quick start" in readme.lower()

    def test_readme_has_reproduction_instructions(self):
        readme = (PKG / "README.md").read_text()
        assert "Reproduction" in readme or "reproduce" in readme.lower()

    def test_requirements_txt_non_empty(self):
        reqs = (PKG / "requirements.txt").read_text().strip()
        assert len(reqs) > 0

    def test_requirements_contains_numpy(self):
        reqs = (PKG / "requirements.txt").read_text().lower()
        assert "numpy" in reqs

    def test_requirements_contains_scikit_learn(self):
        reqs = (PKG / "requirements.txt").read_text().lower()
        assert "scikit" in reqs or "sklearn" in reqs

    def test_setup_py_has_install_requires(self):
        setup = (PKG / "setup.py").read_text()
        assert "install_requires" in setup

    def test_data_manifest_is_valid_json(self):
        data = json.loads((PKG / "data_manifest.json").read_text())
        assert "_description" in data

    def test_model_manifest_is_valid_json(self):
        data = json.loads((PKG / "model_manifest.json").read_text())
        assert "_description" in data

    def test_package_init_has_stage(self):
        import ncaa_submission_2026
        assert hasattr(ncaa_submission_2026, "__stage__")
        assert "09" in ncaa_submission_2026.__stage__


# ---------------------------------------------------------------------------
# run_prediction.py — synthetic mode
# ---------------------------------------------------------------------------

class TestRunPrediction:
    def test_synthetic_run_succeeds(self, tmp_path):
        output = tmp_path / "submission.csv"
        result = subprocess.run(
            [sys.executable, str(PKG / "run_prediction.py"),
             "--synthetic", "--output", str(output)],
            capture_output=True, text=True, cwd=str(ROOT),
        )
        assert result.returncode == 0, (
            f"run_prediction.py failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )

    def test_synthetic_output_file_created(self, tmp_path):
        output = tmp_path / "submission.csv"
        subprocess.run(
            [sys.executable, str(PKG / "run_prediction.py"),
             "--synthetic", "--output", str(output)],
            capture_output=True, cwd=str(ROOT),
        )
        assert output.exists()

    def test_synthetic_output_is_valid_csv(self, tmp_path):
        output = tmp_path / "submission.csv"
        subprocess.run(
            [sys.executable, str(PKG / "run_prediction.py"),
             "--synthetic", "--output", str(output)],
            capture_output=True, cwd=str(ROOT),
        )
        df = pd.read_csv(output)
        assert "ID" in df.columns
        assert "Pred" in df.columns
        assert len(df) > 0

    def test_synthetic_probs_in_valid_range(self, tmp_path):
        output = tmp_path / "submission.csv"
        subprocess.run(
            [sys.executable, str(PKG / "run_prediction.py"),
             "--synthetic", "--output", str(output)],
            capture_output=True, cwd=str(ROOT),
        )
        df = pd.read_csv(output)
        assert (df["Pred"] >= 0.01).all()
        assert (df["Pred"] <= 0.99).all()

    def test_reproducible_outputs_across_runs(self, tmp_path):
        """Two runs with same seed must produce identical CSV."""
        out1 = tmp_path / "sub1.csv"
        out2 = tmp_path / "sub2.csv"
        args = [sys.executable, str(PKG / "run_prediction.py"),
                "--synthetic", "--seed", "42"]
        subprocess.run(args + ["--output", str(out1)], capture_output=True, cwd=str(ROOT))
        subprocess.run(args + ["--output", str(out2)], capture_output=True, cwd=str(ROOT))

        df1 = pd.read_csv(out1).sort_values("ID").reset_index(drop=True)
        df2 = pd.read_csv(out2).sort_values("ID").reset_index(drop=True)
        assert (df1["Pred"].values == df2["Pred"].values).all(), (
            "run_prediction.py produced different outputs across two runs with same seed"
        )


# ---------------------------------------------------------------------------
# Privacy certification
# ---------------------------------------------------------------------------

class TestPrivacyCertification:
    def test_certification_produces_json(self, fitted_model, train_df, val_arrays, tmp_path):
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        X1, X2, y = val_arrays
        certifier = PrivacyCertifier()
        certifier.certify(train_df=train_df, model=fitted_model,
                          X1_val=X1, X2_val=X2)
        certifier.save_json(str(tmp_path / "cert.json"))
        assert (tmp_path / "cert.json").exists()

    def test_certification_json_valid(self, fitted_model, train_df, val_arrays, tmp_path):
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        X1, X2, y = val_arrays
        certifier = PrivacyCertifier()
        certifier.certify(train_df=train_df, model=fitted_model,
                          X1_val=X1, X2_val=X2)
        certifier.save_json(str(tmp_path / "cert.json"))
        data = json.loads((tmp_path / "cert.json").read_text())
        for key in ("passed", "timestamp", "summary", "checks"):
            assert key in data

    def test_certification_json_has_all_check_sections(self, fitted_model, train_df, val_arrays, tmp_path):
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        X1, X2, y = val_arrays
        certifier = PrivacyCertifier()
        certifier.certify(train_df=train_df, model=fitted_model,
                          X1_val=X1, X2_val=X2)
        certifier.save_json(str(tmp_path / "cert.json"))
        data = json.loads((tmp_path / "cert.json").read_text())
        checks = data["checks"]
        for section in ("leakage", "determinism", "network_isolation", "dependency_audit"):
            assert section in checks, f"Missing section: {section}"

    def test_certification_produces_markdown(self, fitted_model, train_df, val_arrays, tmp_path):
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        X1, X2, y = val_arrays
        certifier = PrivacyCertifier()
        certifier.certify(train_df=train_df, model=fitted_model,
                          X1_val=X1, X2_val=X2)
        certifier.save_md(str(tmp_path / "cert.md"))
        assert (tmp_path / "cert.md").exists()

    def test_certification_markdown_has_required_sections(self, fitted_model, train_df, val_arrays, tmp_path):
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        X1, X2, y = val_arrays
        certifier = PrivacyCertifier()
        certifier.certify(train_df=train_df, model=fitted_model,
                          X1_val=X1, X2_val=X2)
        certifier.save_md(str(tmp_path / "cert.md"))
        md = (tmp_path / "cert.md").read_text()
        for section in (
            "Data Leakage",
            "Deterministic Inference",
            "Network Isolation",
            "Dependency Audit",
        ):
            assert section in md, f"Missing MD section: {section}"

    def test_clean_data_certification_passes_all_checks(self, fitted_model, train_df, val_arrays):
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        X1, X2, y = val_arrays
        certifier = PrivacyCertifier()
        report = certifier.certify(train_df=train_df, model=fitted_model,
                                   X1_val=X1, X2_val=X2)
        assert report.passed is True

    def test_leaky_data_certification_fails(self, fitted_model, train_df, val_arrays):
        """Certification should fail when training data contains test-season rows."""
        from ncaa_submission_2026.certification.certifier import PrivacyCertifier
        from ncaa_submission_2026.certification.leakage_detector import DataLeakageDetector

        X1, X2, y = val_arrays
        det = DataLeakageDetector(test_season=2026)
        dirty_train = det.inject_leakage(train_df, n_rows=5, season=2026)

        certifier = PrivacyCertifier(test_season=2026)
        report = certifier.certify(train_df=dirty_train, model=fitted_model,
                                   X1_val=X1, X2_val=X2)
        # Leakage check should fail, causing overall failure
        assert report.leakage.leakage_found is True
        assert report.passed is False


# ---------------------------------------------------------------------------
# Kaggle submission format validation
# ---------------------------------------------------------------------------

class TestKaggleFormat:
    def test_submission_passes_validation(self, fitted_model, feat_df):
        from ncaa_models.baseline import FEATURE_COLS
        from ncaa_models.submit import build_submission, validate_submission
        sub = build_submission(
            model=fitted_model,
            team_features=feat_df,
            team_ids=list(feat_df[feat_df["season"] == 2025]["team_id"].unique()),
            season=2025,
            feature_cols=FEATURE_COLS,
            clip_probs=True,
        )
        result = validate_submission(sub, expected_season=2025)
        assert result["valid"] is True

    def test_no_duplicate_ids(self, fitted_model, feat_df):
        from ncaa_models.baseline import FEATURE_COLS
        from ncaa_models.submit import build_submission
        sub = build_submission(
            model=fitted_model,
            team_features=feat_df,
            team_ids=list(feat_df[feat_df["season"] == 2025]["team_id"].unique()),
            season=2025,
            feature_cols=FEATURE_COLS,
        )
        assert sub["ID"].nunique() == len(sub)

    def test_probs_clipped(self, fitted_model, feat_df):
        from ncaa_models.baseline import FEATURE_COLS
        from ncaa_models.submit import build_submission
        sub = build_submission(
            model=fitted_model,
            team_features=feat_df,
            team_ids=list(feat_df[feat_df["season"] == 2025]["team_id"].unique()),
            season=2025,
            feature_cols=FEATURE_COLS,
            clip_probs=True,
        )
        assert (sub["Pred"] >= 0.01).all()
        assert (sub["Pred"] <= 0.99).all()


# ---------------------------------------------------------------------------
# Dependency auditor
# ---------------------------------------------------------------------------

class TestDependencyAuditor:
    def test_audit_passes(self):
        from ncaa_submission_2026.certification.dependency_auditor import DependencyAuditor
        auditor = DependencyAuditor()
        report = auditor.audit()
        assert report.audit_passed is True

    def test_required_packages_present(self):
        from ncaa_submission_2026.certification.dependency_auditor import (
            DependencyAuditor, REQUIRED_PACKAGES
        )
        auditor = DependencyAuditor()
        report = auditor.audit()
        assert report.missing_required == []

    def test_packages_list_non_empty(self):
        from ncaa_submission_2026.certification.dependency_auditor import DependencyAuditor
        auditor = DependencyAuditor()
        report = auditor.audit()
        assert len(report.packages) > 0

    def test_python_version_recorded(self):
        from ncaa_submission_2026.certification.dependency_auditor import DependencyAuditor
        auditor = DependencyAuditor()
        report = auditor.audit()
        assert "3." in report.python_version


# ---------------------------------------------------------------------------
# Manifest tools
# ---------------------------------------------------------------------------

class TestManifest:
    def test_sha256_bytes(self):
        from ncaa_submission_2026.manifest import sha256_bytes
        h = sha256_bytes(b"hello")
        assert len(h) == 64  # SHA-256 hex = 64 chars
        assert h == sha256_bytes(b"hello")  # deterministic

    def test_sha256_file(self, tmp_path):
        from ncaa_submission_2026.manifest import sha256_file
        f = tmp_path / "test.txt"
        f.write_bytes(b"test content 12345")
        h = sha256_file(f)
        assert len(h) == 64
        assert h == sha256_file(f)

    def test_build_manifest(self, tmp_path):
        from ncaa_submission_2026.manifest import build_manifest
        (tmp_path / "a.txt").write_text("hello")
        (tmp_path / "b.txt").write_text("world")
        manifest = build_manifest(tmp_path)
        assert len(manifest) == 2
        assert all(len(v) == 64 for v in manifest.values())

    def test_build_manifest_empty_dir(self, tmp_path):
        from ncaa_submission_2026.manifest import build_manifest
        empty = tmp_path / "empty"
        empty.mkdir()
        manifest = build_manifest(empty)
        assert manifest == {}

    def test_verify_manifest_all_match(self, tmp_path):
        from ncaa_submission_2026.manifest import build_manifest, verify_manifest
        (tmp_path / "a.txt").write_text("hello")
        manifest = build_manifest(tmp_path)
        results = verify_manifest(tmp_path, manifest)
        assert all(results.values())

    def test_verify_manifest_detects_tamper(self, tmp_path):
        from ncaa_submission_2026.manifest import build_manifest, verify_manifest
        f = tmp_path / "a.txt"
        f.write_text("original")
        manifest = build_manifest(tmp_path)
        f.write_text("tampered!")
        results = verify_manifest(tmp_path, manifest)
        assert not all(results.values())

    def test_save_and_load_manifest(self, tmp_path):
        from ncaa_submission_2026.manifest import (
            build_manifest, save_manifest, load_manifest
        )
        (tmp_path / "data.csv").write_text("id,val\n1,2\n")
        manifest = build_manifest(tmp_path)
        path = tmp_path / "manifest.json"
        save_manifest(manifest, path)
        loaded = load_manifest(path)
        assert loaded == manifest
