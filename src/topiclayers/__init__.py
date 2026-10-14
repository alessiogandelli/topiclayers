"""topiclayers — from posts to topical multilayer networks."""

from importlib.metadata import PackageNotFoundError, version

try:
    from dotenv import load_dotenv

    # Load variables from a .env file (searched from the current working
    # directory upwards). Real environment variables take precedence, so
    # exported secrets are never overridden. Non-fatal if python-dotenv is
    # missing (it is a declared dependency, but keep imports robust).
    load_dotenv(override=False)
except ImportError:
    pass

try:
    __version__ = version("topiclayers")
except PackageNotFoundError:  # running from a source tree without install
    __version__ = "0.0.0"
