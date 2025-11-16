# FieldSense AI v3.0 - Build & Packaging Guide

## Prerequisites

```bash
pip install pyinstaller reportlab
```

## PyInstaller - Standalone Executable

### Build for Current Platform

```bash
# Build using spec file
pyinstaller fieldsense.spec

# Output: dist/fieldsense (or fieldsense.exe on Windows)
```

### Run Executable

```bash
# Linux/Mac
./dist/fieldsense

# Windows
dist\fieldsense.exe
```

### Build Options

```bash
# One-file bundle
pyinstaller --onefile bin/ship.py -n fieldsense

# With icon
pyinstaller --onefile --icon=ui/public/icon.ico bin/ship.py -n fieldsense

# Debug mode
pyinstaller --debug all fieldsense.spec
```

## Tauri - Desktop App

### Setup

```bash
cd ui
npm install
npm install -g @tauri-apps/cli
```

### Build

```bash
# Development
npm run tauri dev

# Production build
npm run tauri build

# Output: ui/src-tauri/target/release/
```

### Platform-Specific Builds

**Windows:**
```bash
npm run tauri build -- --target x86_64-pc-windows-msvc
# Output: .msi installer
```

**macOS:**
```bash
npm run tauri build -- --target x86_64-apple-darwin
# Output: .dmg and .app
```

**Linux:**
```bash
npm run tauri build -- --target x86_64-unknown-linux-gnu
# Output: .deb and .AppImage
```

## Python Package Distribution

### Build Wheel

```bash
python setup.py sdist bdist_wheel

# Output: dist/fieldsense_ai-3.0.0-py3-none-any.whl
```

### Install from Wheel

```bash
pip install dist/fieldsense_ai-3.0.0-py3-none-any.whl
```

### Upload to PyPI

```bash
pip install twine
twine upload dist/*
```

## Docker Container

### Build Image

```bash
docker build -t fieldsense-ai:3.0 .
```

### Run Container

```bash
docker run -p 8080:8080 -p 3000:3000 fieldsense-ai:3.0
```

## Privacy Certification

Generate privacy compliance certificate:

```bash
python -m src.privacy.cert

# Output: certifications/CERT-*.pdf
```

## Testing Build

```bash
# Run full integration test
python bin/ship.py

# Test privacy cert
python -c "from src.privacy import one_click_export; one_click_export()"

# Test package import
python -c "import src; print('Package OK')"
```

## Size Optimization

### Reduce Executable Size

```bash
# Use UPX compression
pyinstaller --upx-dir=/path/to/upx fieldsense.spec

# Exclude unnecessary modules
pyinstaller --exclude-module matplotlib fieldsense.spec
```

### Strip Binaries (Linux)

```bash
strip dist/fieldsense
```

## Cross-Platform Build Matrix

| Platform | Executable | Package | Size |
|----------|-----------|---------|------|
| Windows x64 | fieldsense.exe | .msi | ~200MB |
| macOS x64 | fieldsense.app | .dmg | ~180MB |
| Linux x64 | fieldsense | .AppImage | ~190MB |

## CI/CD Integration

### GitHub Actions

```yaml
name: Build
on: [push, pull_request]
jobs:
  build:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
      - run: pip install -r requirements.txt
      - run: pyinstaller fieldsense.spec
      - uses: actions/upload-artifact@v3
        with:
          name: fieldsense-${{ matrix.os }}
          path: dist/
```

## Troubleshooting

### Missing Dependencies

```bash
# Find hidden imports
pyi-makespec --onefile bin/ship.py
# Add to hiddenimports in .spec file
```

### Large File Size

- Use `--exclude-module` for unused packages
- Enable UPX compression
- Use virtual environment with minimal deps

### Runtime Errors

```bash
# Debug mode
pyinstaller --debug all fieldsense.spec
./dist/fieldsense

# Check imports
pyi-archive_viewer dist/fieldsense
```

## Release Checklist

- [ ] Update version in `setup.py`
- [ ] Update `CHANGELOG.md`
- [ ] Run all tests
- [ ] Generate privacy certificate
- [ ] Build all platforms
- [ ] Test executables
- [ ] Create git tag
- [ ] Create GitHub release
- [ ] Upload artifacts

## Version Info

- **Current Version**: 3.0.0
- **Build Date**: 2025-11-16
- **Python**: >=3.8
- **Platforms**: Windows, macOS, Linux
