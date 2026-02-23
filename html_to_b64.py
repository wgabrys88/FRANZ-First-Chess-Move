import base64
with open("panel_Dual_Epistemology.html", "rb") as f:
    print(base64.b64encode(f.read()).decode("ascii"))