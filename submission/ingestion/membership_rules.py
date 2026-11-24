"""A source-quoted Boolean rule encoded in the existing string value slot.

Leaf conditions remain literal clauses, including geographic references and
exceptions. The tree is a candidate interpretation, not proof of its own logic;
independent semantic source review must establish operators and completeness.
"""
import json
import re


def rule_nodes(value):
    """Return every validated-shape node with a stable child-index path."""
    try:
        root = json.loads(value)
    except (ValueError, TypeError):
        return ()
    result = []

    def visit(node, path=()):
        if not isinstance(node, dict):
            return
        result.append((path, node))
        children = node.get('children')
        if isinstance(children, list):
            for index, child in enumerate(children):
                visit(child, path + (index,))

    visit(root)
    return tuple(result)


def rule_issue(value, *, evidence_text=None, claim_quote=None,
               allow_discontiguous=False):
    try:
        root=json.loads(value)
    except (ValueError,TypeError):
        return 'invalid_rule_json'
    count=0
    leaves=[]
    def visit(node,depth):
        nonlocal count
        count+=1
        if count>64 or depth>8 or not isinstance(node,dict):return 'invalid_rule_structure'
        if set(node)!={'operator','quote','children'}:return 'invalid_rule_fields'
        op,quote,children=node['operator'],node['quote'],node['children']
        if not isinstance(op,str) or op not in {'condition','all','any','except'} or not isinstance(quote,str) or not quote.strip():return 'invalid_rule_node'
        if not isinstance(children,list):return 'invalid_rule_children'
        if evidence_text is not None and quote not in evidence_text:return 'rule_quote_not_in_evidence'
        if op=='condition':
            if children:return 'leaf_has_children'
            leaves.append(quote)
            return None
        if len(children)<2:return 'rule_operator_requires_children'
        marker={'all':r'\b(?:and|all|each|both)\b','any':r'\b(?:or|any|either)\b','except':r'\b(?:except|unless|but not)\b'}[op]
        if not re.search(marker,quote,re.I):return 'rule_operator_not_explicit'
        spans=[]
        for child in children:
            error=visit(child,depth+1)
            if error:return error
            if child['quote'] not in quote:
                if not allow_discontiguous:
                    return 'child_outside_parent_rule'
                # In a multi-Block rule, the parent quote carries the explicit
                # Boolean operator while child quotes are separately source
                # bound. A caller must verify every node against its Evidence.
                if evidence_text is not None and child['quote'] not in evidence_text:
                    return 'rule_quote_not_in_evidence'
                continue
            start=quote.index(child['quote']);end=start+len(child['quote'])
            if any(start<b and a<end for a,b in spans):return 'overlapping_rule_siblings'
            spans.append((start,end))
        if len({c['quote'] for c in children})!=len(children):return 'duplicate_rule_condition'
        return None
    issue=visit(root,0)
    if issue:return issue
    # A quoted amount/age/threshold cannot disappear from the interpreted leaves.
    # Link destinations and ordered-list markers are syntax, not constraints.
    from collections import Counter
    def numbers(text):
        text=re.sub(r'\[([^\]]+)\]\([^\n]*?\)',r'\1',text)
        text=re.sub(r'(?m)^\s*\d+[.)]\s*','',text)
        return Counter(re.findall(r'\b\d+(?:[,.]\d+)*\b',text))
    remaining=numbers(root['quote'])
    for leaf in leaves:remaining.subtract(numbers(leaf))
    if any(n>0 for n in remaining.values()):return 'rule_omits_numeric_condition'
    if claim_quote is not None and root['quote']!=claim_quote:return 'rule_claim_mismatch'
    return None


def eligibility_benefit(text, context=''):
    plain=re.sub(r'[*_]', '',text)
    explicit=bool(re.search(r'\b(?:live or work|immediate family member|eligible to join|membership required)\b',plain,re.I))
    grouped=bool(re.search(r'\b(?:eligible|join|membership)\b',context,re.I)
                 and re.search(r'^\s*[-+]?\s*(?:You|Applicants?|Members?)\s+(?:live|work|have|belong|must)',plain,re.I))
    if (explicit or grouped) and re.search(r'\b(?:and|or|if|unless|except|any|all|family|charter|must|belong)\b',plain,re.I):
        return 'eligibility_rule_structure'
    return None


def rate_benefit(text):
    if (re.search(r'\b(?:APR|annual percentage rate)\b',text,re.I)
            and re.search(r'\d(?:\.\d+)?\s*%',text)
            and re.search(r'\b(?:introductory|standard|promotional|revert|after)\b',text,re.I)):
        return 'rate_disclosure_binding'
    return None
