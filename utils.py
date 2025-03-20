import re
import pandas


def clean_html_table(html):
    """ Remove the first empty <th> in <thead> and 
    remove all the <th> from the lines of <tbody>. 
    """
    html = re.sub(r'<th class="blank level0" >&nbsp;</th>\s*', '', html, count=1)
    html = re.sub(r'<th(?!.*?col_heading).*?</th>', '', html, flags=re.DOTALL)

    return html


def color_gradient(val):
    """Generate a color from red to green or grey for NaN values.
    """
    if pandas.isna(val):
        return "background-color: #D3D3D3; color: black; text-align: center;"
    if isinstance(val, (int, float)):
        color = f"rgba({255 - int(val * 255)}, {int(val * 255)}, 100, 0.6)"
        return f"background-color: {color}; color: black; text-align: center;"
    return ""


def fix_html(html):
    """Fix pyvis updateFilter function by removing string 
    varibale access to enable filtering in gradio"""
    pattern = r"(function updateFilter\s*\(.*?\)\s*\{)"

    new_functions = """
    function updateFilterProperty(value) { 
        filter["property"] = value;
    }
    function updateFilterItem(value) { 
        filter["item"] = value;
    }
    """
    fixed_html = html.replace("'", "\"")
    fixed_html = re.sub(pattern, new_functions + r"\n\1", fixed_html, flags=re.DOTALL)
    fixed_html = fixed_html.replace("updateFilter(value, \"item\")",
                                          "updateFilterItem(value)")
    fixed_html = fixed_html.replace("updateFilter(value, \"property\")",
                                          "updateFilterProperty(value)")

    return fixed_html

def display_metrics(metrics_df, model):
    """Return the HTML code to display the computed metrics in a table"""
    styled_html = clean_html_table(metrics_df.to_html(index=None))
    return f"""
    <div style="width:100%; overflow-x: auto;">
        <h2>{model} Results</h2>
        <style>
            table {{
                width: 100%;
                font-size: 18px;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 12px;
                text-align: center;
                border: 1px solid #ddd;
            }}
        </style>
        {styled_html}
    </div>
    """