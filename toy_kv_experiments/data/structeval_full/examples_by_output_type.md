# StructEval Examples by Output Type

## Angular

- `task_id`: `000100`
- `task_name`: `Text to Angular`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Angular code, not HTML code. (Define the component's template and styles inline, Angular code should be self-contained in a single file, do not use any external dependencies or libraries, for example, do not use functionalities of zone.js):

Task:
Transform the given text input into an Angular component that presents a detailed user report view.

Feature Requirements:
- Include a centered header using an <h1> element displaying the text "User Report Dashboard".
- Create a <div> container with a class name of 'report-wrapper' to encapsulate all component content.
- Construct exactly 2 distinct sections: the first for user statistics and the second for activity logs.
- The first section must render a table (<table>) element featuring exactly 2 rows and 3 columns; ensure the header row uses <th> elements to label each column.
- Within the table, assign a CSS class named 'status-critical' to one specific cell to highlight an important statistic in bold text.
- The second section should display a paragraph (<p>) with an inline style that sets the text color to green and the font style to italic.
- Position a button labeled "Update Report" at the bottom using a <div> with a class 'button-area' for alignment.
```

### Raw Output Metric

```json
[
  "User Report Dashboard",
  "report-wrapper",
  "status-critical",
  "Update Report",
  "button-area",
  "<h1>",
  "<div class=\"report-wrapper\">",
  "<table>",
  "<th>",
  "<p>",
  "color: green",
  "font-style: italic",
  "{{"
]
```

## Latex

- `task_id`: `00XX20`
- `task_name`: `Text to Latex`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Latex:

Task:
Convert the given plain text description of a statistical concept into a LaTeX formatted representation of a probability density function for the normal distribution.

Feature Requirements:
- The generated expression must begin with \begin{equation} and end with \end{equation}.
- Use the \frac{}{} command to clearly format the coefficient preceding the exponential term.
- Format the exponential part using the \exp command, ensuring that its exponent is enclosed in curly braces.
- Represent the constants π and the square root using the \pi and \sqrt{} commands, respectively.
- All variables, such as x, μ, and σ, must be italicized using standard LaTeX math mode.
- Employ parentheses where necessary to clarify the order of operations, especially around the squared term.
- Insert appropriate spacing commands (e.g., \,, \; or \;) between different elements of the equation.
- Ensure that the denominator is clearly expressed as a product involving 2 and σ.
```

### Raw Output Metric

```json
[
  "\\begin{equation}",
  "\\end{equation}",
  "\\frac",
  "\\exp",
  "\\pi",
  "\\sqrt",
  "x",
  "\\mu",
  "\\sigma",
  "(x-\\mu)^2",
  "2\\sigma",
  "\\,",
  "\\;"
]
```

## Markdown

- `task_id`: `000700`
- `task_name`: `Text to Markdown`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Markdown:

Task:
Create a formatted Markdown document outlining a project update meeting agenda for a new product launch.

Feature Requirements:
- Begin with a centered title using an <h1> element that displays the text "Project Update Meeting".
- Include an <h2> element titled "Meeting Agenda" immediately after the title.
- Present exactly 3 bullet points under the "Meeting Agenda" section, where each bullet point briefly describes a specific agenda item.
- Add an introductory section with a single paragraph containing exactly 3 sentences that provides background information for the meeting.
- Include a "Discussion Points" section using an <h2> element, followed by exactly 4 paragraphs where each paragraph is comprised of 2 full sentences discussing different aspects of the project.
- Create a "Next Steps" section using an <h2> element that is immediately followed by a numbered list containing exactly 3 items, with each item detailing a clear and distinct action.
- Conclude with a "Summary" section using an <h2> element, featuring one final paragraph of exactly 2 sentences that encapsulate the outcomes of the meeting.
```

### Raw Output Metric

```json
[
  "# Project Update Meeting",
  "## Meeting Agenda",
  "## Discussion Points",
  "## Next Steps",
  "## Summary",
  "1.",
  "2.",
  "3."
]
```

## Matplotlib

- `task_id`: `000802`
- `task_name`: `Text to Matplotlib`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Matplotlib:

Task:
Generate a Matplotlib scatter plot using predefined data arrays representing wind speed and precipitation levels.

Feature Requirements:
- Display a scatter plot with the x-axis labeled as "Wind Speed (mph)" and the y-axis labeled as "Precipitation (mm)".
- Include a title at the top of the plot with the text "Wind Speed vs Precipitation".
- Plot exactly 8 data points representing different measurement pairs.
- Use red circle markers with a marker size of 80 for each data point.
- Set the x-axis range from 0 to 100 and the y-axis range from 0 to 50.
- Add grid lines to both the x and y axes.
- Include a legend positioned in the lower right corner with the label "Measurement Data".
```

### Raw Output Metric

```json
[
  "Wind Speed vs Precipitation",
  "Wind Speed (mph)",
  "Precipitation (mm)",
  "Measurement Data",
  "plt.scatter",
  "red",
  "marker='o'",
  "s=80",
  "plt.xlim(0, 100)",
  "plt.ylim(0, 50)",
  "grid",
  "lower right",
  "plt.xlabel",
  "put.ylabel",
  "put.title"
]
```

## React

- `task_id`: `001140`
- `task_name`: `Text to React`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output React:

Task:
Generate a React component that displays a basic newsletter subscription form.

Feature Requirements:
- Include a centered header using an <h1> element with the text "Subscribe to our Newsletter".
- Display two text input fields placed side by side: one with a placeholder "Enter First Name" and the other with a placeholder "Enter Email Address".
- Wrap the input fields in a <form> element with an appropriate onSubmit handler.
- Provide exactly one button labeled "Subscribe" styled with a green background color.
- Use a flexbox layout to center the form horizontally on the page.
- Implement a <div> element below the form that shows the message "Subscription Successful" when the "Subscribe" button is clicked.
- Add basic client-side validation to ensure both input fields are not empty before processing the form submission.
```

### Raw Output Metric

```json
[
  "Subscribe to our Newsletter",
  "Enter First Name",
  "Enter Email Address",
  "Subscribe",
  "Subscription Successful",
  "h1",
  "form",
  "button",
  "div",
  "backgroundColor: \"green\"",
  "onSubmit",
  "flex",
  "input"
]
```

## SVG

- `task_id`: `001200`
- `task_name`: `Text to SVG`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output SVG:

Task:
Generate an SVG image of a simple house with a roof, door, windows, and a sun in the background.

Feature Requirements:
- The SVG canvas must have a width and height of 400 pixels.
- The house body should be a square with a width and height of 200 pixels, centered horizontally and positioned so that its bottom is 50 pixels above the bottom of the canvas.
- The house body must be filled with light gray (#D3D3D3) and include a black border stroke of 2 pixels.
- The roof should be an isosceles triangle placed on top of the house body, with a base width of 220 pixels and a height of 80 pixels, filled with dark red (#8B0000) and centered above the house.
- A rectangular door must be placed at the center of the house body’s bottom, with a width of 50 pixels and a height of 80 pixels, filled with brown (#A0522D).
- Two square windows, each 40 pixels by 40 pixels, must be positioned on either side of the door on the house body, filled with white and outlined with a black stroke of 2 pixels.
- A circular sun should be positioned in the top right corner of the canvas with a radius of 30 pixels, filled with yellow (#FFD700) and outlined in black with a stroke of 2 pixels.
- All elements must be clearly separated and proportionally aligned, ensuring no overlapping between the house and the sun.
```

### Raw Output Metric

```json
[
  "<svg",
  "width=\"400\"",
  "height=\"400\"",
  "<rect",
  "fill=\"#D3D3D3\"",
  "stroke=\"black\"",
  "stroke-width=\"2\"",
  "200",
  "50",
  "<polygon",
  "fill=\"#8B0000\"",
  "220",
  "80",
  "fill=\"#A0522D\"",
  "fill=\"white\"",
  "40",
  "<circle",
  "r=\"30\"",
  "fill=\"#FFD700\"",
  "</svg>"
]
```

## Tikz

- `task_id`: `001302`
- `task_name`: `Text to Tikz`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Tikz:

Task:
Generate a Tikz diagram representing a star-inspired node network.

Feature Requirements:
- Display exactly 8 nodes, one of which is a central node and the remaining 7 evenly arranged in a circle around it.
- Label the central node with "Center" and the peripheral nodes with letters A through G.
- Position the central node at the exact center of a canvas measuring 12 cm by 12 cm, with peripheral nodes evenly spaced around a circle of radius 4 cm.
- Draw the central node as a blue circle with a radius of 0.5 cm, and each peripheral node as a red circle with a radius of 0.3 cm.
- Connect the central node to each peripheral node using dashed lines with a thickness of 1mm.
- Ensure all nodes are evenly distributed with precise angular spacing between the peripheral nodes.
- Place a title "Star Network Diagram" centered above the diagram in an italic font.
- Use solid fill for nodes and no external assets or icons.
- Guarantee that every line and node is explicitly placed based on the described measurements and positions.
```

### Raw Output Metric

```json
[
  "Star Network Diagram",
  "Center",
  "A",
  "B",
  "C",
  "D",
  "E",
  "F",
  "G",
  "blue",
  "red",
  "circle",
  "dashed"
]
```

## HTML

- `task_id`: `000400`
- `task_name`: `Text to HTML`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output HTML:

Task:
Generate an HTML page that transforms a short narrative about a notable scientific discovery into a structured document.

Feature Requirements:
- Include a main title using an <h1> element with the text "Discovery Overview".
- Display the provided narrative inside a <p> element immediately following the title.
- Insert a horizontal rule using the <hr> element after the paragraph to separate content sections.
- Add a subheading with an <h2> element titled "Event Details".
- Create an ordered list using an <ol> element containing exactly 3 items: "Date of Discovery", "Location", and "Research Team".
- Place each list item in an <li> element in the order specified.
- Conclude the HTML structure with a <footer> element that contains the text "Document End".
```

### Raw Output Metric

```json
[
  "Discovery Overview",
  "Event Details",
  "Date of Discovery",
  "Location",
  "Research Team",
  "Document End",
  "<h1>",
  "<p>",
  "<hr>",
  "<h2>",
  "<ol>",
  "<li>Date of Discovery</li>",
  "<li>Location</li>",
  "<li>Research Team</li>",
  "<footer>"
]
```

## Mermaid

- `task_id`: `000903`
- `task_name`: `Text to Mermaid`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Mermaid:

Task:
Convert a textual description of an organization hierarchy into a Mermaid flowchart diagram.

Feature Requirements:
- Include a title at the top of the flowchart labeled "Organization Hierarchy".
- Display exactly 4 main branches representing distinct departments, each within its own node, arranged horizontally.
- Use a circular-shaped node to depict the CEO at the top of the hierarchy.
- Connect all nodes with arrows to represent direct reporting lines.
- Include at least one subnode under each department to illustrate the position of a team lead.
- Utilize rectangular nodes exclusively for department head positions.
- Apply a unique fill color for each department branch to differentiate them visually.
- Incorporate a loopback arrow on one department branch to indicate a recurring reporting cycle.
- Arrange the diagram in a top-down hierarchical layout with clear directional flow.
- Include a legend at the bottom of the diagram that clearly explains the meaning of all node shapes and colors used.
```

### Raw Output Metric

```json
[
  "graph TD",
  "Organization Hierarchy",
  "((CEO))",
  "[Department 1]",
  "[Department 2]",
  "[Department 3]",
  "[Department 4]",
  "Team Lead",
  "-->",
  "style",
  "fill:#",
  "legend"
]
```

## Typst

- `task_id`: `001401`
- `task_name`: `Text to Typst`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Typst code that can be directly compiled by Typst CLI compiler to PDF (Use markdown-like typst syntax style, do not have any advanced usage such as variables):

Task:
Convert a brief text description into a Typst document representing a sample magazine layout.

Feature Requirements:
- Display a main title using a level-1 heading with the text "Magazine Feature" at the top of the document.
- Create a section with a level-2 heading titled "Article Summary" followed by a paragraph containing placeholder text.
- Add two subsections under "Article Summary": one with a level-3 heading "Interview" and the other with a level-3 heading "Insights", each followed by a list of exactly four bullet points with placeholder content.
- Insert a horizontal line between the two subsections to separate them visually.
- Create a secondary section with a level-2 heading "Additional Details" that includes two paragraphs of different placeholder text.
- Format the content so that each bullet point in the lists is indented uniformly beneath its respective heading.
- Place a centered divider at the bottom of the page using a horizontal line.
- Include a footer area at the very end with a smaller font size containing centered text reading "© 2023 Magazine".
```

### Raw Output Metric

```json
[
  "Magazine Feature",
  "Article Summary",
  "Interview",
  "Insights",
  "Additional Details",
  "© 2023 Magazine"
]
```

## Vega

- `task_id`: `001502`
- `task_name`: `Text to Vega`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Vega:

Task:
Create a Vega specification for a line chart visualization tracking the monthly average temperatures in several fictional cities.

Feature Requirements:
- Include a title with the text "Fictional City Temperature Trends" displayed at the top of the visualization.
- Plot exactly 3 distinct lines, each representing a different fictional city.
- Assign a unique color to each line, ensuring no two lines share the same color.
- Label the x-axis with the months of the year, ensuring that each month is fully visible and evenly spaced.
- Set the y-axis to display temperature values in Celsius, with tick marks at every 5-degree increment.
- Place a distinct marker (e.g., a circle) on every data point along each line to emphasize the exact monthly values.
- Include a legend on the right side of the chart that maps each line color to its corresponding fictional city.
- Overlay horizontal grid lines corresponding to each y-axis tick to improve readability.
- Display numerical temperature values next to each data point for precise value representation.
```

### Raw Output Metric

```json
[
  "Fictional City Temperature Trends"
]
```

## Vue

- `task_id`: `001600`
- `task_name`: `Text to Vue`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Vue code in the format of Single File Components (SFC):

Task:
Convert a plain text event schedule into a Vue component that displays a timeline of events.

Feature Requirements:
- Render a header using an <h2> element with the text "Event Timeline" at the top of the component.
- Create an array in the data property containing event objects, each with a title and a time property.
- Use a v-for loop to iterate over the events array and display each event inside a <div> container.
- Each <div> should include a <p> element for the event title and a <span> element for the event time.
- Apply a CSS class named "event-item" to each event container for styling.
- Render a button with the text "Remove Event" inside each event container that, when clicked, removes that specific event from the list.
- Provide an <input> element with the placeholder "Enter event title" and another with the placeholder "Enter event time" above the list.
- Include an "Add Event" button next to the inputs that appends a new event, using the values from the input fields, to the events array.
- Define a computed property that keeps track of and displays the total number of events currently in the schedule.
```

### Raw Output Metric

```json
[
  "Event Timeline",
  "event-item",
  "Remove Event",
  "Enter event title",
  "Enter event time",
  "Add Event"
]
```

## TikZ

- `task_id`: `08XX20`
- `task_name`: `Convert Matplotlib to TikZ`
- `input_type`: `Matplotlib`
- `rendering`: `True`

### Query Preview

```text
Please output TikZ:

Task:
Convert the following Matplotlib code to TikZ code.

<code>
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------
# SECTION 1: Data Preparation
# -----------------------------------------------------
np.random.seed(42)

# Generate smooth data for line plots
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)
noise = np.random.normal(scale=0.2, size=100)
y_noisy = np.sin(x) + noise

# Generate random data for scatter plot
x_scatter = np.random.rand(50) * 10
y_scatter = np.sin(x_scatter) + np.random.normal(scale=0.3, size=50)

# Data for bar chart, comparing two categories across some groups
categories = ['A', 'B', 'C', 'D']
values_cat1 = [5, 7, 3, 4]
values_cat2 = [6, 9, 5, 2]
index = np.arange(len(categories))
bar_width = 0.35

# -----------------------------------------------------
# SECTION 2: Initialize the Figure and Layout
# -----------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Complex Matplotlib Visualization", fontsize=16, fontweight='bold')

# -----------------------------------------------------
# Plot 1: Trigonometric Functions
# Description:
# - Plots sin(x) in blue 
# - Plots cos(x) in orange with dashed lines
# -----------------------------------------------------
ax = axs[0, 0]
ax.plot(x, y_sin, label='sin(x)', color='blue', linewidth=2)
ax.plot(x, y_cos, label='cos(x)', color='orange', linestyle='--', linewidth=2)
ax.set_title("Trigonometric Functions")
ax.set_xlabel("X values")
ax.set_ylabel("Function Value
```

### Raw Output Metric

```json
[
  "Complex Matplotlib Visualization",
  "Trigonometric Functions",
  "X values",
  "Function Values",
  "sin(x)",
  "cos(x)",
  "Noisy Sine Wave",
  "Amplitude",
  "Clean sin(x)",
  "Noisy sin(x)"
]
```

## Canvas

- `task_id`: `000300`
- `task_name`: `Text to Canvas`
- `input_type`: `Text`
- `rendering`: `True`

### Query Preview

```text
Please output Canvas:

Task:
Render a digital recipe card entirely using a canvas element for everything visible.

Feature Requirements:
- Create a single canvas element with dimensions of 800x600 pixels.
- Display a centered title "Grandma's Apple Pie" at the top using a bold 28px font.
- Draw a straight line under the title that stretches from the left margin to the right margin using canvas drawing methods.
- Divide the canvas into two horizontal sections; the upper section for ingredients and the lower section for cooking steps.
- In the upper section, list 4 ingredients with each item separated by a marker (e.g., a dash) using a 20px regular font.
- In the lower section, display 3 sequential cooking steps numbered 1 to 3 using a 20px italic font.
- Position the ingredients on the left side occupying 40% of the width and the cooking steps on the right side occupying the remaining 60%.
- At the bottom of the canvas, draw a horizontal separator with canvas drawing commands to emphasize the conclusion of the content.
- Use only canvas drawing functions such as fillText, moveTo, and lineTo to render all text and lines.
```

### Raw Output Metric

```json
[
  "800",
  "600",
  "Grandma's Apple Pie",
  "fillText",
  "moveTo",
  "lineTo",
  "-",
  "1",
  "2",
  "3"
]
```

## JSON

- `task_id`: `000500`
- `task_name`: `Text to JSON`
- `input_type`: `Text`
- `rendering`: `False`

### Query Preview

```text
Please output JSON code:

Task:
Given a summary of a fictional novel, extract a structured representation of its key metadata, main characters, and chapter details.

Feature Requirements:
1. The field title under novel represents the novel's full title as a string.
2. The field name inside novel.author indicates the author's name as a string.
3. The field birth_year inside novel.author specifies the author's year of birth as a four-digit integer.
4. The field year under novel.publication gives the year the novel was published as a four-digit integer.
5. The field publisher within novel.publication provides the name of the publishing house as a string.
6. The field genres inside novel is a list of strings, each representing a literary genre associated with the novel.
7. The name field within the first element of novel.characters list represents the primary character's name as a string.
8. The role field inside the first element of novel.characters specifies the character's narrative role (e.g., protagonist, antagonist) as a string.
9. The traits field in the first element of novel.characters is a list of strings describing key personality traits.
10. The name field within the second element of novel.characters contains another main character's name as a string.
11. The role field in the second element of novel.characters specifies this character's narrative role as a string.
12. The traits field in the second element of novel.characters is a list of strings enumerating this character's personality traits.
13. The title field inside the first element of novel.chapters gives t
```

### Raw Output Metric

```json
[
  "novel.title",
  "novel.author.name",
  "novel.author.birth_year",
  "novel.publication.year",
  "novel.publication.publisher",
  "novel.genres",
  "novel.characters[0].name",
  "novel.characters[0].role",
  "novel.characters[0].traits",
  "novel.characters[1].name",
  "novel.characters[1].role",
  "novel.characters[1].traits",
  "novel.chapters[0].title",
  "novel.chapters[0].word_count",
  "novel.chapters[0].events",
  "novel.chapters[1].title",
  "novel.chapters[1].word_count"
]
```

## CSV

- `task_id`: `000200`
- `task_name`: `Text to CSV`
- `input_type`: `Text`
- `rendering`: `False`

### Query Preview

```text
Please output CSV code(Only output the CSV content as plain text, starting with the header row.):

Task:
Given a textual description of a unique recipe, generate a CSV row representing its core attributes, ingredients, steps, and metadata.

Feature Requirements:
1. The recipe_id field must be a unique identifier for each recipe, represented as a string or integer.
2. The recipe_name field should contain the full name of the recipe as a string.
3. The chef_name field must include the name of the person or entity who created the recipe, as a string.
4. The cuisine_type field should specify the type of cuisine (such as Italian, Thai, etc.), formatted as a string.
5. The prep_time_minutes field must indicate the preparation time in minutes, as an integer.
6. The cook_time_minutes field should capture the cooking time in minutes, as an integer.
7. The difficulty_level field must state the recipe's difficulty, chosen from a set list (such as 'Easy', 'Medium', 'Hard') as a string.
8. The ingredient_list field should be a semicolon-separated list of ingredient names, each as a string, within a single CSV cell.
9. The ingredient_quantities field must be a semicolon-separated list of quantities corresponding to each ingredient, as numbers, within a single CSV cell.
10. The ingredient_units field should be a semicolon-separated list of units (such as grams, cups), each as a string, matching the order of ingredient_list, within a single CSV cell.
11. The step_descriptions field must provide a semicolon-separated list of instructions for each step, as strings, within a single CSV cell.

```

### Raw Output Metric

```json
[
  "csv::recipe_id",
  "csv::recipe_name",
  "csv::chef_name",
  "csv::cuisine_type",
  "csv::prep_time_minutes",
  "csv::cook_time_minutes",
  "csv::difficulty_level",
  "csv::ingredient_list",
  "csv::ingredient_quantities",
  "csv::ingredient_units",
  "csv::step_descriptions",
  "csv::step_timers",
  "csv::servings",
  "csv::vegetarian",
  "csv::allergen_warnings",
  "csv::rating_average",
  "csv::review_count"
]
```

## TOML

- `task_id`: `001000`
- `task_name`: `Text to TOML`
- `input_type`: `Text`
- `rendering`: `False`

### Query Preview

```text
Please output TOML code:

Task:
Given a description of a fictional museum, produce a TOML document representing its structure, including details about its director, location, galleries, and artworks.

Feature Requirements:
1. The key museum.name is a string representing the official name of the museum, located at the top level inside the museum object.
2. The key museum.location.city is a string specifying the city where the museum is located, inside the location object within museum.
3. The key museum.location.country is a string indicating the country of the museum, found within the location object inside museum.
4. The key museum.founded_year is an integer representing the year the museum was established, included directly under museum.
5. The key museum.director.full_name is a string containing the full name of the museum's director, within the director object under museum.
6. The key museum.director.tenure_years is an integer showing the number of years the director has held their position, inside director within museum.
7. The key museum.galleries[0].title is a string for the title of the first gallery, inside the first element of the galleries list under museum.
8. The key museum.galleries[0].floor is an integer specifying which floor the first gallery is located on, within the first galleries element under museum.
9. The key museum.galleries[0].artworks[0].title is a string representing the title of the first artwork in the first gallery, inside the first element of the artworks list within the first galleries element.
10. The key museum.galleries[0].artworks[0].art
```

### Raw Output Metric

```json
[
  "museum.name",
  "museum.location.city",
  "museum.location.country",
  "museum.founded_year",
  "museum.director.full_name",
  "museum.director.tenure_years",
  "museum.galleries[0].title",
  "museum.galleries[0].floor",
  "museum.galleries[0].artworks[0].title",
  "museum.galleries[0].artworks[0].artist",
  "museum.galleries[0].artworks[0].year_created",
  "museum.galleries[0].artworks[0].medium",
  "museum.galleries[0].artworks[0].dimensions.height_cm",
  "museum.galleries[0].artworks[0].dimensions.width_cm",
  "museum.galleries[0].artworks[0].on_display"
]
```

## XML

- `task_id`: `001700`
- `task_name`: `Text to XML`
- `input_type`: `Text`
- `rendering`: `False`

### Query Preview

```text
Please output XML code:

Task:
Generate an XML representation of a creative recipe, including all details about the recipe, its author, ingredients, preparation steps, nutritional information, tags, and metadata.

Feature Requirements:
1. The title element within the recipe node should provide the name of the recipe as a string.
2. The name field inside the author object, which is nested within recipe, must be a string representing the author's full name.
3. The email field within the contact object, itself inside author under recipe, should be a string formatted as a valid email address.
4. Each name field inside the ingredients list, which is part of recipe, must be a string naming the ingredient.
5. Each quantity in the ingredients list inside recipe should be a number indicating the amount of the ingredient.
6. Each unit field in the ingredients list within recipe must be a string specifying the measurement unit, such as 'grams' or 'cups'.
7. The order field within each step object in the steps list under recipe must be an integer indicating the sequence number of the step.
8. The instruction field inside each step in the steps list under recipe should be a string describing the action to take.
9. The minutes field within the duration object inside each step under steps in recipe must be an integer for the time in minutes required for that step.
10. The calories field within the nutrition object under recipe should be a number representing the total calories per serving.
11. The grams field inside the fat object within nutrition under recipe must be a number indicating 
```

### Raw Output Metric

```json
[
  "recipe.title",
  "recipe.author.name",
  "recipe.author.contact.email",
  "recipe.ingredients.*.name",
  "recipe.ingredients.*.quantity",
  "recipe.ingredients.*.unit",
  "recipe.steps.*.order",
  "recipe.steps.*.instruction",
  "recipe.steps.*.duration.minutes",
  "recipe.nutrition.calories",
  "recipe.nutrition.fat.grams",
  "recipe.nutrition.protein.grams",
  "recipe.nutrition.carbohydrates.grams",
  "recipe.tags.*",
  "recipe.metadata.created_at"
]
```

## YAML

- `task_id`: `001800`
- `task_name`: `Text to YAML`
- `input_type`: `Text`
- `rendering`: `False`

### Query Preview

```text
Please output YAML code:

Task:
Describe a fictional creature, including its key characteristics, abilities, and discovery details.

Feature Requirements:
1. The field name under creature represents the creature's given name as a string.
2. The field species within creature specifies the biological or fantastical classification as a string.
3. The field type inside creature.habitat indicates the primary environment where the creature lives, as a string value such as forest, ocean, or desert.
4. The field region within creature.habitat provides a string identifying the specific area or locale of the habitat.
5. The field climate inside creature.habitat describes the general weather conditions of the habitat as a string (e.g., tropical, arid).
6. The field name inside the first element of the creature.abilities list gives the name of the creature's primary ability as a string.
7. The field description within the first element of creature.abilities is a string explaining what the ability does.
8. The field cooldown_seconds in the first element of creature.abilities specifies the cooldown time in seconds as an integer.
9. The field name inside the second element of creature.abilities provides the name of another ability as a string.
10. The field description within the second element of creature.abilities is a string detailing how the second ability functions.
11. The field cooldown_seconds in the second element of creature.abilities indicates the cooldown period in seconds as an integer.
12. The field size_meters under creature.stats gives the creature's length or height as a 
```

### Raw Output Metric

```json
[
  "creature.name",
  "creature.species",
  "creature.habitat.type",
  "creature.habitat.region",
  "creature.habitat.climate",
  "creature.abilities[0].name",
  "creature.abilities[0].description",
  "creature.abilities[0].cooldown_seconds",
  "creature.abilities[1].name",
  "creature.abilities[1].description",
  "creature.abilities[1].cooldown_seconds",
  "creature.stats.size_meters",
  "creature.stats.weight_kg",
  "creature.stats.lifespan_years",
  "creature.stats.diet",
  "creature.discovery.discovered_by",
  "creature.discovery.discovery_date"
]
```
