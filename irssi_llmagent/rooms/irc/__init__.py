"""
IRC-specific functionality for irssi integration.
"""

from .varlink import VarlinkClient, VarlinkSender

__all__ = ["VarlinkClient", "VarlinkSender"]
