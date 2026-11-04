"""Writes lora_trained.drawio: what the v5.5 CLIP model trains (page 1: whole model, frozen vs trained; page 2: one
LoRA-adapted layer, W frozen plus the A/B side path). Colours match vlm_pipeline.drawio's legend.
Run: python docs/vlm/diagrams/make_lora_trained.py
"""
from html import escape
from pathlib import Path

STYLE = {
    "input": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#d5e8d4;strokeColor=#82b366;",
    "op": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#dae8fc;strokeColor=#6c8ebf;",
    "frozen": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fillColor=#eeeeee;strokeColor=#666666;",
    "train": "rounded=1;whiteSpace=wrap;html=1;fontSize=12;fontStyle=1;fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;",
    "group": "rounded=1;whiteSpace=wrap;html=1;fontSize=13;fontStyle=1;fillColor=none;dashed=1;strokeColor=#999999;"
             "verticalAlign=top;spacingTop=4;",
    "text": "text;html=1;fontSize=12;align=center;verticalAlign=middle;",
}
EDGE = "edgeStyle=orthogonalEdgeStyle;rounded=1;html=1;strokeColor=#555555;endArrow=classic;"


class Page:
    def __init__(self, name):
        self.name, self.cells, self.n = name, [], 0

    def box(self, value, style, x, y, w, h, extra=""):
        self.n += 1
        cid = f"v{self.n}"
        self.cells.append(f'<mxCell id="{cid}" value="{escape(value)}" style="{STYLE.get(style, style)}{extra}" '
                          f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def edge(self, src, dst, exit_=None, entry=None):
        self.n += 1
        pts = ""
        if exit_:
            pts += f"exitX={exit_[0]};exitY={exit_[1]};exitDx=0;exitDy=0;"
        if entry:
            pts += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
        self.cells.append(f'<mxCell id="e{self.n}" style="{EDGE}{pts}" edge="1" parent="1" source="{src}" target="{dst}">'
                          f'<mxGeometry relative="1" as="geometry"/></mxCell>')

    def xml(self):
        return (f'<diagram name="{escape(self.name)}"><mxGraphModel grid="1"><root><mxCell id="0"/><mxCell id="1" parent="0"/>'
                + "".join(self.cells) + "</root></mxGraphModel></diagram>")


def legend(p, y):
    p.box("Frozen (pretrained CLIP)", "frozen", 120, y, 200, 30)
    p.box("Trained", "train", 360, y, 200, 30)


def whole_model():
    p = Page("What is trained")
    photo = p.box("<b>Plant photo</b><br>224 px, augmented", "input", 50, 40, 260, 58)
    p.box("a photo of", "frozen", 370, 40, 90, 26)
    p.box("[crop]", "train", 465, 40, 70, 26)
    p.box("leaves with", "frozen", 540, 40, 90, 26)
    p.box("[disease]", "train", 370, 72, 90, 26)
    desc = p.box(": symptom description", "frozen", 465, 72, 165, 26)

    img = p.box("CLIP image encoder (ViT-B/16)", "group", 50, 120, 260, 275)
    txt = p.box("CLIP text encoder (all frozen)", "group", 370, 120, 260, 275)
    for i in range(12):
        y = 156 + 19 * i
        if i < 4:
            p.box(f"Block {i + 1}", "frozen", 66, y, 228, 16, "align=left;spacingLeft=6;fontSize=11;")
        else:
            p.box(f"Block {i + 1}", "frozen", 66, y, 170, 16, "align=left;spacingLeft=6;fontSize=11;")
            p.box("LoRA", "train", 242, y, 52, 16, "fontSize=11;")
        p.box(f"Block {i + 1}", "frozen", 386, y, 228, 16, "fontSize=11;")

    cos = p.box("<b>Cosine similarity</b><br>image vs each of 333 pair prompts", "op", 160, 430, 360, 56)
    p.edge(photo, img)
    p.edge(desc, txt, exit_=(0.2121, 1), entry=(0.5, 0))  # x = 500, under the prompt
    p.edge(img, cos, exit_=(0.5, 1), entry=(0.0556, 0))
    p.edge(txt, cos, exit_=(0.5, 1), entry=(0.944, 0))
    p.box("Trained: 55 crop + 175 disease vectors; LoRA A/B in blocks 5-12 (~1.57M). Everything else frozen.",
          "text", 50, 556, 580, 24)
    legend(p, 516)
    return p


def one_layer():
    p = Page("Inside one LoRA layer")
    x = p.box("<b>Input</b><br>768 numbers", "input", 40, 188, 90, 56)
    w = p.box("<b>W (frozen)</b><br>3072 x 768<br>2.36M numbers", "frozen", 210, 50, 140, 140)
    a = p.box("A", "shape=trapezoid;perimeter=trapezoidPerimeter;direction=south;size=0.4;html=1;fontSize=13;fontStyle=1;"
                   "fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;", 210, 256, 52, 88)
    b = p.box("B", "shape=trapezoid;perimeter=trapezoidPerimeter;direction=north;size=0.4;html=1;fontSize=13;fontStyle=1;"
                   "fillColor=#e1d5e7;strokeColor=#9673a6;strokeWidth=2;", 298, 256, 52, 88)
    p.box("LoRA (trained)", "text", 210, 226, 140, 20)
    p.box("768 -> 16 -> 3072", "text", 210, 352, 140, 20)
    plus = p.box("+", "ellipse;whiteSpace=wrap;html=1;fontSize=16;fontStyle=1;fillColor=#eeeeee;strokeColor=#666666;",
                 414, 200, 32, 32)
    out = p.box("<b>Output</b><br>3072 numbers", "op", 490, 188, 140, 56)
    p.box("A: 16 x 768<br>B: 3072 x 16<br>61k trained numbers<br>B starts at 0", "text", 490, 270, 140, 70)
    p.edge(x, w, exit_=(1, 0.5), entry=(0, 0.5))
    p.edge(x, a, exit_=(1, 0.5), entry=(0, 0.5))
    p.edge(a, b, exit_=(1, 0.5), entry=(0, 0.5))
    p.edge(w, plus, exit_=(1, 0.5), entry=(0.5, 0))
    p.edge(b, plus, exit_=(1, 0.5), entry=(0.5, 1))
    p.edge(plus, out, exit_=(1, 0.5), entry=(0, 0.5))
    p.box("Output = W·x + B·(A·x) = (W + B·A)·x", "text", 40, 390, 590, 24)
    legend(p, 430)
    return p


out = Path(__file__).with_name("lora_trained.drawio")
out.write_text('<mxfile host="drawio">' + whole_model().xml() + one_layer().xml() + "</mxfile>", encoding="utf-8")
print(f"wrote {out}")
