"""Tree Edit Distance based Similarity (TEDS) metric for table evaluation.

Based on IBM Research's TEDS implementation (Apache 2.0 License).
Reference: Zhong et al., "Image-based Table Recognition: Data, Model, and Evaluation", 2019.
Original: https://github.com/ibm-aur-nlp/PubTabNet

TEDS compares HTML table structures using APTED (All-Path Tree Edit Distance).
Score ranges from 0 to 1, where 1 = perfect match.

Formula:
    TEDS = 1.0 - (tree_edit_distance / max(nodes_pred, nodes_gt))
"""

from collections import deque

import Levenshtein
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html


class TableTree(Tree):
    """Tree node for APTED comparison, storing HTML table cell attributes."""

    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        if self.tag == "td":
            result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                self.tag, self.colspan, self.rowspan, self.content,
            )
        else:
            result = '"tag": %s' % self.tag
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)


class CustomConfig(Config):
    """APTED config for table tree comparison.

    Node rename cost:
    - Different tag/colspan/rowspan → 1.0
    - Same tag, td cells → normalized Levenshtein distance on content
    - Same tag, non-td → 0.0 (free rename)
    """

    @staticmethod
    def maximum(*sequences):
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        if (node1.tag != node2.tag
                or node1.colspan != node2.colspan
                or node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance based Similarity for table evaluation.

    Args:
        structure_only: If True, ignore cell content (compare structure only).
            This gives TEDS-S score.
        ignore_nodes: List of HTML tags to strip before comparison.
    """

    def __init__(self, structure_only: bool = False, ignore_nodes: list[str] | None = None):
        self.structure_only = structure_only
        self.ignore_nodes = ignore_nodes
        self._tokens: list[str] = []

    def _tokenize(self, node) -> None:
        """Tokenize HTML node content into character-level tokens."""
        self._tokens.append("<%s>" % node.tag)
        if node.text is not None:
            self._tokens += list(node.text)
        for n in node.getchildren():
            self._tokenize(n)
        if node.tag != "unk":
            self._tokens.append("</%s>" % node.tag)
        if node.tag != "td" and node.tail is not None:
            self._tokens += list(node.tail)

    def _load_html_tree(self, node, parent=None) -> TableTree | None:
        """Convert lxml HTML node to TableTree for APTED."""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self._tokens = []
                self._tokenize(node)
                cell = self._tokens[1:-1].copy()
            new_node = TableTree(
                node.tag,
                int(node.attrib.get("colspan", "1")),
                int(node.attrib.get("rowspan", "1")),
                cell,
                *deque(),
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self._load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred_html: str, gt_html: str) -> float:
        """Compute TEDS score between predicted and ground truth HTML tables.

        Args:
            pred_html: Predicted HTML table string.
            gt_html: Ground truth HTML table string.

        Returns:
            TEDS score in [0, 1]. 1.0 = perfect match.
        """
        if not pred_html or not gt_html:
            return 0.0

        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred_html, parser=parser)
        true = html.fromstring(gt_html, parser=parser)

        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]

            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)

            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)

            if n_nodes == 0:
                return 0.0

            tree_pred = self._load_html_tree(pred)
            tree_true = self._load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def calculate_teds(pred_html: str, gt_html: str, structure_only: bool = False) -> float:
    """Calculate TEDS score between predicted and ground truth HTML tables.

    Args:
        pred_html: Predicted HTML table.
        gt_html: Ground truth HTML table.
        structure_only: If True, return TEDS-S (structure only, ignoring content).

    Returns:
        TEDS score in [0, 1] (1.0 = perfect match).
    """
    return TEDS(structure_only=structure_only).evaluate(pred_html, gt_html)


def calculate_teds_s(pred_html: str, gt_html: str) -> float:
    """Calculate TEDS-S (structure-only) score."""
    return TEDS(structure_only=True).evaluate(pred_html, gt_html)
