# Prompt Templates

This directory contains Jinja2 templates for the benchmark agent prompts.

## Templates

- `system_prompt.j2` - System prompt that defines the agent's behavior, workflow, boundaries, and tool usage guidelines
- `user_message.j2` - User message template that presents the specific problem instance to solve

## Template Variables

### system_prompt.j2

Currently takes no variables (static template). The system prompt defines the general behavior and guidelines for all problem instances.

### user_message.j2

Available variables for rendering the user message:

| Variable | Type | Description | Example |
|----------|------|-------------|---------|
| `repo` | string | Repository name | `"django/django"` |
| `base_commit` | string | Git commit hash (optional) | `"abc123def..."` |
| `problem` | string | Problem statement (multiline) | Full issue description |
| `fail_to_pass` | string | Tests that must pass after fix | `'["test_foo.py::test_bar"]'` |
| `hints_text` | string | Optional hints (conditional) | Additional guidance text |

**Conditional Rendering:**
- `base_commit` section only shows if value is provided
- `hints_text` section only shows if value is provided

## Customization

To customize prompts:

1. **Copy this templates directory** to your desired location:
   ```bash
   cp -r cc_benchmark/templates /path/to/custom/templates
   ```

2. **Modify the template files** as needed:
   - Edit `system_prompt.j2` to change agent behavior, add/remove instructions
   - Edit `user_message.j2` to change how problems are presented

3. **Use custom templates** in your code:
   ```python
   from pathlib import Path
   from cc_benchmark.prompts import PromptManager

   manager = PromptManager(template_dir=Path("/path/to/custom/templates"))
   system_prompt = manager.render_system_prompt()
   user_message = manager.render_user_message(
       repo="owner/repo",
       problem="Description...",
       fail_to_pass='["test.py::test_func"]',
       base_commit="abc123",
       hints_text="",
   )
   ```

## Inline Fallbacks

If template files are not found or cannot be loaded, the system automatically falls back to inline templates defined in `cc_benchmark/prompts.py`. This ensures the system works even without template files in place.

## Template Syntax

Templates use **Jinja2** syntax with a **sandboxed environment** for security:
- `{{ variable }}` - Insert variable value
- `{% if condition %}...{% endif %}` - Conditional blocks
- `{% for item in items %}...{% endfor %}` - Loops (if needed)
- `{%- ... -%}` - Trim whitespace

**Security Note:** The system uses `SandboxedEnvironment` to prevent code injection, even if user-provided content (like problem statements) contains Jinja2 syntax.

## Best Practices

1. **Keep templates focused**: System prompt should be general, user message should be specific
2. **Test changes**: Render templates with real data before deploying
3. **Version control**: Track template changes in git to see evolution over time
4. **Document modifications**: Add comments in templates explaining customizations
5. **Validate output**: Ensure rendered prompts don't exceed token limits

## Examples

### Minimal System Prompt

```jinja2
You are a software engineer fixing bugs.

Follow these steps:
1. Analyze the problem
2. Locate the code
3. Implement a fix
4. Test it works

Modify source files only, not tests or config.
```

### Extended User Message

```jinja2
# Bug Report: {{ repo }}

## Problem
{{ problem }}

## Required Tests
{{ fail_to_pass }}

{% if hints_text %}
## Additional Context
{{ hints_text }}
{% endif %}

Fix this issue by modifying source code only.
```

## Troubleshooting

**Templates not loading?**
- Check that template directory path is correct
- Verify files exist and have `.j2` extension
- System will fall back to inline templates silently

**Rendering errors?**
- Check variable names match exactly
- Ensure all required variables are provided
- Look for syntax errors in Jinja2 templates

**Variables not appearing?**
- Verify variable is passed to `render_user_message()`
- Check conditional blocks (`{% if %}`) have correct logic
- Ensure no trailing whitespace issues (`{%-` trims whitespace)
