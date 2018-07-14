--- layout: default title: Science and Spontaneity --- <style>body { background-color: #ffe4e1 }</style>

# {{ page.title }}

## Upcoming Blog Posts/Future Ideas

<font size="3">*   My leg injury, and why self reported symptoms can be dangerous*   I enjoy sunny weather: Biomechanisms behind humans' universal love for the sun</font>

<font size="3"><font>

<font size="4">{% for post in site.posts %}*   <span>{{ post.date | date_to_string }}</span> : [{{ post.title }}]({{ post.url }} "{{ post.title }}"){% endfor %}</font>

</font></font>