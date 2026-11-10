"""
Parsel lets you extract text from XML/HTML documents using XPath
or CSS selectors
"""

__author__ = "Scrapy project"
__email__ = "info@scrapy.org"
__version__ = "1.12.1"
__all__ = [
    "AbsoluteXPathWarning",
    "DocumentProvenance",
    "LocatedNode",
    "NodeKind",
    "Provenance",
    "Selector",
    "SelectorList",
    "StaleProvenanceError",
    "css2xpath",
    "xpathfuncs",
]

from parsel import xpathfuncs
from parsel.csstranslator import css2xpath
from parsel.provenance import (
    DocumentProvenance,
    LocatedNode,
    NodeKind,
    Provenance,
    StaleProvenanceError,
)
from parsel.selector import AbsoluteXPathWarning, Selector, SelectorList

xpathfuncs.setup()
