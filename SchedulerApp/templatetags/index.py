from django import template
register = template.Library()

@register.filter
def dictKey(d, k):
    '''Returns the given key from a dictionary.'''

    return ', '.join(d[k]) or ''


@register.simple_tag
def sub(s, d, w, t, break_times):
    '''Returns the subject-teacher for a department, weekday, and time period or indicates a break.'''
    # Check if the current time and weekday is a break
    if (t, w) in break_times:
        return 'Break'  # Return 'Break' if the time is a break

    # If not a break, look for the class
    for c in s:
        if c.department.dept_name.strip().lower() == d.strip().lower() and c.meeting_time.day == w and c.meeting_time.time == t:
            return f'{c.course.course_name} ({c.instructor.name})'
    
    return 'Free'  # Return 'Free' if no match is found






@register.tag
def active(parser, token):
    args = token.split_contents()
    template_tag = args[0]
    if len(args) < 2:
        raise (template.TemplateSyntaxError, f'{template_tag} tag requires at least one argument')
    return NavSelectedNode(args[1:])

class NavSelectedNode(template.Node):
    def __init__(self, patterns):
        self.patterns = patterns
    def render(self, context):
        path = context['request'].path
        for p in self.patterns:
            pValue = template.Variable(p).resolve(context)
            if path == pValue:
                return 'active'
        return ''