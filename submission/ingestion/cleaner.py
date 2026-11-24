"""Conservative Markdown cleaning before semantic sectioning."""

import html
import re
from collections.abc import Callable
from urllib.parse import urlsplit

from ingestion.markdown_tables import is_table_start, table_end


class PageCleaner:
    ESCAPED_CRLF = re.compile(r"(?<!\\)" + re.escape(r"\\r\\n"))
    PROTECTED_INLINE = re.compile(
        r"`+[^`\n]*`+|!?\[(?:\\.|[^\]\\])*\]\((?:\\.|[^()\\]|\([^()]*\))*\)"
        r"|https?://\S+"
    )

    @classmethod
    def _map_unprotected(cls, text: str, transform: Callable[[str], str]) -> str:
        """Apply a transform outside inline code, Markdown links, and URLs."""
        parts: list[str] = []
        start = 0
        for match in cls.PROTECTED_INLINE.finditer(text):
            parts.extend([transform(text[start:match.start()]), match.group(0)])
            start = match.end()
        parts.append(transform(text[start:]))
        return "".join(parts)

    @classmethod
    def _normalize_extraction_artifacts(cls, markdown: str) -> str:
        """Repair the observed BEL and double-escaped CRLF outside code/links."""
        output: list[str] = []
        fence_char = None
        fence_length = 0
        for line in markdown.splitlines():
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if fence_char is not None or fence:
                output.append(line)
                if fence_char is None:
                    fence_char, fence_length = fence[1][0], len(fence[1])
                elif (fence and fence[1][0] == fence_char
                      and len(fence[1]) >= fence_length and not fence[2].strip()):
                    fence_char = None
                continue

            def normalize(text: str) -> str:
                return cls.ESCAPED_CRLF.sub("\n", text.replace("\x07", " "))

            output.append(cls._map_unprotected(line, normalize))
        return "\n".join(output)

    CAPTCHA_IMAGE = re.compile(r"!\[\s*\]\((https?://[^\s)]+)\)")
    EXTERNAL_LINK_NOTICE = (
        "We only do this when it's helpful for you. But we must inform you that "
        "Dupaco isn't responsible for the site's content, products, services, policies "
        "or sponsors. Also, Dupaco's Privacy Policy does not apply to third-party sites. "
        "So, if you have concerns, please look at its privacy disclosures."
    )
    CAPTCHA_CONTROLS = {
        "recaptcha", "recaptcha requires verification.", "protected by **recaptcha**",
        "please try again.", "please select all matching images.",
        "please also check the new images.",
        "please select around the object, or reload if there are none.", "skip", "verify",
    }
    PLUGIN_UI_LABELS = ("messenger launcher", "messenger")
    CAPTCHA_PROMPT = re.compile(
        r"Select all (?:images|squares) with .+ (?:Click verify once there are none left\.?|If there are none, click skip)",
        re.IGNORECASE,
    )

    @classmethod
    def is_captcha_table(cls, text: str) -> bool:
        images = list(cls.CAPTCHA_IMAGE.finditer(text))
        if not images:
            return False
        for image in images:
            try:
                url = urlsplit(image.group(1))
            except ValueError:
                return False
            if (url.hostname not in {"google.com", "www.google.com"}
                    or url.path != "/recaptcha/enterprise/payload"):
                return False
        return bool(re.fullmatch(r"[\s|:\-]*", cls.CAPTCHA_IMAGE.sub("", text)))

    @classmethod
    def _remove_verified_ui(cls, lines: list[str]) -> list[str]:
        """Remove exact UI templates while preserving mixed/business content."""
        output: list[str] = []
        fence_char = None
        fence_length = 0
        captcha_context = False
        i = 0
        while i < len(lines):
            line = lines[i]
            text = line.strip()
            fence = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
            if fence_char is not None or fence:
                output.append(line)
                if fence_char is None:
                    fence_char, fence_length = fence[1][0], len(fence[1])
                elif (fence and fence[1][0] == fence_char
                      and len(fence[1]) >= fence_length and not fence[2].strip()):
                    fence_char = None
                i += 1
                continue
            if text == cls.EXTERNAL_LINK_NOTICE or text.strip("*").casefold() == "heads up! this link leads to a different website.":
                i += 1
                continue
            if text.casefold() == "third party plugins":
                end = i + 1
                matched = True
                for label in cls.PLUGIN_UI_LABELS:
                    while end < len(lines) and not lines[end].strip():
                        end += 1
                    if end >= len(lines) or lines[end].strip().casefold() != label:
                        matched = False
                        break
                    end += 1
                if matched:
                    i = end
                    continue
            if text.casefold() == "recaptcha":
                captcha_context = True
                i += 1
                continue
            if captcha_context and (
                text.casefold() in cls.CAPTCHA_CONTROLS
                or cls.CAPTCHA_PROMPT.fullmatch(text)
            ):
                i += 1
                continue
            if is_table_start(lines, i):
                end = table_end(lines, i)
                table = "\n".join(lines[i:end])
                if cls.is_captcha_table(table):
                    captcha_context = True
                    i = end
                    continue
                output.extend(lines[i:end])
                captcha_context = False
                i = end
                continue
            if text:
                captcha_context = False
            output.append(line)
            i += 1
        return output

    # Repeated substantive/promotional blocks remain in source text. Deduplicate
    # their retrieval entries later while retaining every source occurrence.
    RECOMMENDATION_HEADINGS = {
        "recent news",
        "handpicked for you",
        "related articles",
        "you may also like",
        "you may also want to consider",
    }

    NOISE_LINES = {
        "back to top",
        "✕",
        "×",
    }

    HEADING = re.compile(r"^(#{1,6})\s+(.+?)(?:\s+#+)?\s*$")
    # Markdown destinations can contain escaped parentheses or balanced pairs.
    # Avoid a greedy wildcard that could consume prose or additional links.
    _LINK_DESTINATION = r"\((?:\\.|[^()\\\n]|\((?:\\.|[^()\\\n])*\))*\)"
    MODAL_CONTROLS = re.compile(
        r"\[continue\]" + _LINK_DESTINATION
        + r"\s+(?:cancel|\[cancel\]" + _LINK_DESTINATION + r")",
        re.IGNORECASE,
    )
    IMAGE = re.compile(r"!\[((?:\\.|[^\]\\])*)\]\((?:\\.|[^\\\n])*\)")
    TEXT_ENTITIES = re.compile(
        r"&(?:amp;)*(?:#(?:[0-9]+|[xX][0-9a-fA-F]+)|nbsp|quot|apos);"
    )
    TYPOGRAPHY = str.maketrans({
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
        "\u00a0": " ", "\u202f": " ", "\u00ad": "",
    })

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        """Normalize inline typography/entities while protecting code and links."""
        def decode(match: re.Match) -> str:
            original = match.group(0)
            value = original
            for _ in range(3):
                decoded = html.unescape(value)
                if decoded == value:
                    break
                value = decoded
            # Do not introduce Markdown delimiters or control characters from
            # escaped source text (notably pipes inside table cells).
            if any(char in "|<>`\\\n\r" or not char.isprintable()
                   for char in value if char not in "\u00a0\u202f\u00ad"):
                return original
            return value

        return cls._map_unprotected(
            text,
            lambda value: cls.TEXT_ENTITIES.sub(decode, value).translate(cls.TYPOGRAPHY),
        )

    RECOMMENDATIONS_START = "<!-- cleaner:recommendations:start -->"
    RECOMMENDATIONS_END = "<!-- cleaner:recommendations:end -->"
    # Verified malformed hierarchy in the supplied snapshots, keyed by the
    # page's H1 and exact body heading following "Handpicked for You".
    RECOMMENDATION_BOUNDARY_REPAIRS = {
        "recreational loans for your favorite toys": {"the fine print"},
        "the perks of belonging": {"the fine print"},
        "find your kind of credit card": {"the fine print"},
        "your home for business real estate loans": {"your loan details"},
        "your life, fully funded": {"member perks"},
        "build a life worth loving, together": {"let’s talk"},
    }

    @staticmethod
    def _image_description(match: re.Match) -> str:
        """Preserve all nonempty alt text, including short numeric descriptions."""
        return match.group(1).strip()

    @staticmethod
    def _heading_title(title: str) -> str:
        # Normalize presentation only for matching; preserve the original output.
        title = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", title)
        title = title.replace("*", "").replace("_", "")
        return " ".join(title.casefold().split()).rstrip(":")

    def clean(self, markdown: str) -> str:
        markdown = self._normalize_extraction_artifacts(markdown)
        lines = self._remove_verified_ui(markdown.splitlines())
        output = []
        in_recommendations = False
        recommendation_level = 0
        recommendation_title = None
        document_title = None
        fence_char = None
        fence_length = 0

        for line in lines:
            stripped = line.strip()
            # Code fence protection
            fence = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
            if fence_char is not None:
                output.append(line.rstrip())
                if (fence and fence.group(1)[0] == fence_char
                        and len(fence.group(1)) >= fence_length
                        and not fence.group(2).strip()):
                    fence_char = None
                continue
            if fence:
                fence_char = fence.group(1)[0]
                fence_length = len(fence.group(1))
                output.append(line.rstrip())
                continue

            # Rebuild our own boundaries on repeat cleaning rather than nest them.
            if stripped in {self.RECOMMENDATIONS_START, self.RECOMMENDATIONS_END}:
                continue

            line = self._normalize_text(line)
            stripped = line.strip()
            heading = self.HEADING.match(stripped)

            if heading:
                level = len(heading.group(1))
                title = self._heading_title(heading.group(2))
                if level == 1 and document_title is None:
                    document_title = title
                is_recommendation = title in self.RECOMMENDATION_HEADINGS
                boundary_repair = (
                    recommendation_title == "handpicked for you"
                    and title in self.RECOMMENDATION_BOUNDARY_REPAIRS.get(
                        document_title, set()
                    )
                )
                # Heading hierarchy is authoritative except for the verified
                # page-specific extraction defects listed above.
                if in_recommendations and (
                    level <= recommendation_level or boundary_repair
                ):
                    output.extend(["", self.RECOMMENDATIONS_END, ""])
                    in_recommendations = False
                    recommendation_title = None
                    recommendation_level = 0
                if is_recommendation:
                    output.extend(["", self.RECOMMENDATIONS_START])
                    in_recommendations = True
                    recommendation_level = level
                    recommendation_title = title

            if stripped.casefold() in self.NOISE_LINES:
                continue

            # A whole-line dialog control, not a sentence mentioning cancellation.
            if self.MODAL_CONTROLS.fullmatch(stripped):
                continue

            # Standalone descriptive images may contain otherwise missing facts.
            image = self.IMAGE.fullmatch(stripped)
            if image:
                description = self._image_description(image)
                if description:
                    output.append(f"Image description: {description}")
                continue

            # Normalize whitespace-only lines so blank-line collapsing is stable.
            output.append(line.rstrip() if stripped else "")

        if in_recommendations:
            output.extend(["", self.RECOMMENDATIONS_END])

        cleaned = "\n".join(output)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
