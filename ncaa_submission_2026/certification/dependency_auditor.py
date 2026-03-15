"""
ncaa_submission_2026.certification.dependency_auditor
======================================================
Stage 09 — Python dependency audit.

Lists all installed packages with exact versions and verifies that no
non-public (private/internal) packages are required for inference.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Packages required for core inference (must be present)
REQUIRED_PACKAGES: List[str] = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
]

# Packages that are optional (warn if missing but don't fail)
OPTIONAL_PACKAGES: List[str] = [
    "torch",
    "onnxruntime",
    "requests",
]

# Known private/non-public package name patterns (lowercase)
_PRIVATE_PATTERNS: List[str] = [
    "internal",
    "private",
    "corp",
    "enterprise",
    "proprietary",
]


@dataclass
class PackageInfo:
    """Information about one installed package."""

    name: str
    version: str
    is_public: bool = True
    location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "is_public": self.is_public,
            "location": self.location,
        }


@dataclass
class DependencyAuditReport:
    """Result of a dependency audit."""

    python_version: str
    packages: List[PackageInfo] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    private_packages: List[str] = field(default_factory=list)
    audit_passed: bool = True
    details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "python_version": self.python_version,
            "packages": [p.to_dict() for p in self.packages],
            "missing_required": self.missing_required,
            "private_packages": self.private_packages,
            "audit_passed": self.audit_passed,
            "details": self.details,
        }


class DependencyAuditor:
    """
    Audits the Python environment for inference readiness.

    Parameters
    ----------
    required : list of str, optional
        Package names that must be present.
    private_patterns : list of str, optional
        Substrings that indicate a non-public package (case-insensitive).
    """

    def __init__(
        self,
        required: Optional[List[str]] = None,
        private_patterns: Optional[List[str]] = None,
    ) -> None:
        self.required = list(required or REQUIRED_PACKAGES)
        self.private_patterns = list(private_patterns or _PRIVATE_PATTERNS)

    def audit(self) -> DependencyAuditReport:
        """
        Perform a full dependency audit.

        Returns
        -------
        DependencyAuditReport
        """
        packages = self._list_installed()
        installed_names: Set[str] = {p.name.lower() for p in packages}

        # Check required packages
        missing = [
            req for req in self.required
            if req.lower() not in installed_names
            and req.lower().replace("-", "_") not in installed_names
            and req.lower().replace("_", "-") not in installed_names
        ]

        # Check for private packages
        private = [
            p.name for p in packages
            if any(pat in p.name.lower() for pat in self.private_patterns)
        ]

        audit_passed = len(missing) == 0
        detail_parts = []
        if missing:
            detail_parts.append(f"Missing required packages: {missing}")
        if private:
            detail_parts.append(f"Potential private packages: {private}")
        if audit_passed and not private:
            detail_parts.append(
                f"All {len(self.required)} required packages present. "
                f"{len(packages)} packages installed total."
            )

        report = DependencyAuditReport(
            python_version=sys.version,
            packages=packages,
            missing_required=missing,
            private_packages=private,
            audit_passed=audit_passed,
            details=" | ".join(detail_parts),
        )

        if audit_passed:
            logger.info("Dependency audit PASSED: %s", report.details)
        else:
            logger.warning("Dependency audit issues: %s", report.details)

        return report

    def requirements_txt(self) -> str:
        """Generate a pinned requirements.txt string from installed packages."""
        packages = self._list_installed()
        lines = sorted(
            f"{p.name}=={p.version}"
            for p in packages
            if p.version and p.version != "unknown"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _list_installed(self) -> List[PackageInfo]:
        """Return a list of all installed packages."""
        packages: List[PackageInfo] = []
        try:
            import importlib.metadata as meta
            for dist in meta.distributions():
                name = dist.metadata.get("Name", "unknown")
                version = dist.metadata.get("Version", "unknown")
                location = str(getattr(dist, "_path", ""))
                is_pub = not any(pat in name.lower() for pat in self.private_patterns)
                packages.append(PackageInfo(
                    name=name,
                    version=version,
                    is_public=is_pub,
                    location=location,
                ))
        except Exception as e:
            logger.warning("Could not list packages via importlib.metadata: %s", e)
            # Fallback via pkg_resources
            try:
                import pkg_resources
                for dist in pkg_resources.working_set:
                    name = dist.project_name
                    version = dist.version
                    is_pub = not any(pat in name.lower() for pat in self.private_patterns)
                    packages.append(PackageInfo(name=name, version=version, is_public=is_pub))
            except Exception as e2:
                logger.warning("Fallback package listing also failed: %s", e2)
        return packages
