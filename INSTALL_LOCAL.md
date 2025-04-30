# Local Installation

This guide explains how to install the OpenServ SDK locally for development and testing.

## Install from local source

You can install the package directly from your local directory:

```bash
pip install -e .
```

This creates an "editable" installation, where changes to the source code will be immediately reflected in your environment.

## Install from wheel file

Alternatively, you can build and install the wheel file:

```bash
# Build the wheel
python -m build

# Install the wheel
pip install dist/openserv_sdk-0.1.0-py3-none-any.whl
```

## Usage after installation

After installing, you can import and use the package:

```python
import openserv

# Create agent instance
agent = openserv.Agent(...)
``` 