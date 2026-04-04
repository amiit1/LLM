# Lemmatization vs Stemming

Lemmatization and stemming both reduce words to a base form, but they work differently:

- Lemmatization is context-aware and dictionary-based, so it usually returns valid words.
- Stemming is rule-based and often truncates words, which can produce non-dictionary outputs.

| Word         | Lemmatized   | Stemmed |
| ------------ | ------------ | ------- |
| running      | run          | run     |
| studies      | study        | studi   |
| better       | good         | better  |
| caring       | care         | care    |
| leaves       | leaf         | leav    |
| ponies       | pony         | poni    |
| driving      | drive        | drive   |
| organization | organization | organ   |
| happiness    | happiness    | happi   |
| went         | go           | went    |

## Quick Takeaway

- Use lemmatization when you need linguistically accurate root words.
- Use stemming when speed is more important than perfect linguistic quality.
