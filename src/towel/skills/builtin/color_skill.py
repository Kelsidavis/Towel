"""Color skill — convert between color formats, generate palettes."""

from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition


def _hex_to_rgb(h: str) -> tuple[int,int,int]:
    h = h.lstrip("#")
    if len(h) == 3: h = h[0]*2 + h[1]*2 + h[2]*2
    return int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

def _rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"

def _rgb_to_hsl(r: int, g: int, b: int) -> tuple[int,int,int]:
    r2,g2,b2 = r/255, g/255, b/255
    mx,mn = max(r2,g2,b2), min(r2,g2,b2)
    l = (mx+mn)/2
    if mx == mn: h=s=0
    else:
        d = mx-mn
        s = d/(2-mx-mn) if l>0.5 else d/(mx+mn)
        if mx==r2: h=(g2-b2)/d+(6 if g2<b2 else 0)
        elif mx==g2: h=(b2-r2)/d+2
        else: h=(r2-g2)/d+4
        h/=6
    return int(h*360), int(s*100), int(l*100)


class ColorSkill(Skill):
    @property
    def name(self) -> str: return "color"
    @property
    def description(self) -> str: return "Convert colors between hex/RGB/HSL, generate palettes"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="color_convert", description="Convert a color between hex, RGB, and HSL formats",
                parameters={"type":"object","properties":{
                    "color":{"type":"string","description":"Color value (e.g., #ff6600, rgb(255,102,0), red)"},
                },"required":["color"]}),
            ToolDefinition(name="color_palette", description="Generate a color palette (complementary, analogous, triadic)",
                parameters={"type":"object","properties":{
                    "base":{"type":"string","description":"Base color (hex)"},
                    "type":{"type":"string","enum":["complementary","analogous","triadic","shades"],"description":"Palette type"},
                },"required":["base"]}),
            ToolDefinition(name="color_contrast", description="Check contrast ratio between two colors (WCAG accessibility)",
                parameters={"type":"object","properties":{
                    "color1":{"type":"string","description":"First color (hex)"},
                    "color2":{"type":"string","description":"Second color (hex)"},
                },"required":["color1","color2"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "color_convert": return self._convert(arguments["color"])
            case "color_palette": return self._palette(arguments["base"], arguments.get("type","complementary"))
            case "color_contrast": return self._contrast(arguments["color1"], arguments["color2"])
            case _: return f"Unknown tool: {tool_name}"

    def _convert(self, color: str) -> str:
        named = {"red":"#ff0000","green":"#00ff00","blue":"#0000ff","white":"#ffffff",
                 "black":"#000000","yellow":"#ffff00","cyan":"#00ffff","magenta":"#ff00ff",
                 "orange":"#ff8000","purple":"#800080","pink":"#ffc0cb","gray":"#808080"}
        c = color.strip().lower()
        if c in named: c = named[c]
        if c.startswith("#"):
            r,g,b = _hex_to_rgb(c)
        elif c.startswith("rgb"):
            import re
            m = re.findall(r"\d+", c)
            r,g,b = int(m[0]),int(m[1]),int(m[2])
        else:
            return f"Cannot parse color: {color}"
        h,s,l = _rgb_to_hsl(r,g,b)
        hx = _rgb_to_hex(r,g,b)
        return f"Color: {hx}\n  RGB: rgb({r}, {g}, {b})\n  HSL: hsl({h}, {s}%, {l}%)\n  Preview: ████ (see hex in your editor)"

    def _palette(self, base: str, ptype: str) -> str:
        r,g,b = _hex_to_rgb(base)
        h,s,l = _rgb_to_hsl(r,g,b)
        colors = [base]
        if ptype == "complementary":
            colors.append(_hsl_to_hex((h+180)%360, s, l))
        elif ptype == "analogous":
            colors.extend([_hsl_to_hex((h+30)%360,s,l), _hsl_to_hex((h-30)%360,s,l)])
        elif ptype == "triadic":
            colors.extend([_hsl_to_hex((h+120)%360,s,l), _hsl_to_hex((h+240)%360,s,l)])
        elif ptype == "shades":
            for lv in [max(l-20,0), max(l-10,0), min(l+10,100), min(l+20,100)]:
                colors.append(_hsl_to_hex(h, s, lv))
        return f"Palette ({ptype}):\n" + "\n".join(f"  {c}" for c in colors)

    def _contrast(self, c1: str, c2: str) -> str:
        def lum(r,g,b):
            def ch(v):
                v=v/255
                return v/12.92 if v<=0.03928 else ((v+0.055)/1.055)**2.4
            return 0.2126*ch(r)+0.7152*ch(g)+0.0722*ch(b)
        l1 = lum(*_hex_to_rgb(c1))
        l2 = lum(*_hex_to_rgb(c2))
        lighter = max(l1,l2)
        darker = min(l1,l2)
        ratio = (lighter + 0.05) / (darker + 0.05)
        aa_normal = "PASS" if ratio >= 4.5 else "FAIL"
        aa_large = "PASS" if ratio >= 3.0 else "FAIL"
        aaa = "PASS" if ratio >= 7.0 else "FAIL"
        return (f"Contrast ratio: {ratio:.2f}:1\n"
                f"  WCAG AA (normal text): {aa_normal}\n"
                f"  WCAG AA (large text):  {aa_large}\n"
                f"  WCAG AAA:              {aaa}")


def _hsl_to_hex(h: int, s: int, l: int) -> str:
    s2,l2 = s/100, l/100
    c = (1-abs(2*l2-1))*s2
    x = c*(1-abs((h/60)%2-1))
    m = l2-c/2
    if h<60: r1,g1,b1=c,x,0
    elif h<120: r1,g1,b1=x,c,0
    elif h<180: r1,g1,b1=0,c,x
    elif h<240: r1,g1,b1=0,x,c
    elif h<300: r1,g1,b1=x,0,c
    else: r1,g1,b1=c,0,x
    return _rgb_to_hex(int((r1+m)*255),int((g1+m)*255),int((b1+m)*255))
