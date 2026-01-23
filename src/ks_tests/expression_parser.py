import re

# Regex pattern that covers all cases

def parse_expression(expr):
    match = re.match(r"^\s*(-?\d+)?\s*([\+\-]?\d*\*?)?\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$", expr)
    if match:
        a = match.group(1)  # Constant term
        b = match.group(2)  # Coefficient + sign
        dist = match.group(3)  # Distribution name

        # Convert `a` if present, otherwise default to 0
        a = int(a) if a else 0

        # Handle `b` (check if it's empty, and set to 1 or -1 based on the sign)
        if b:
            b = b.replace("*", "").strip()  # Remove `*`
            if b == "" or b == "+":
                b = 1
            elif b == "-":
                b = -1
            else:
                b = int(b)  # Convert to integer
        else:
            b = 1  # Default coefficient is `1` if no coefficient is provided

        return (a, b, dist)

    return None  # No match found
