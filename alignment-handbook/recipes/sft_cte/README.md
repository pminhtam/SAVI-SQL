# Scripts to Train and Evaluate Chat Models

This contain config for text-to-sql 

This idea is train small LLM in 2 role : CTE generator and SQL generator

## Data format 

Data have format like this:
```json
[
    {
        "role": "user",
        "content": "prompt_template.format(dbschema_text=dbschema_text)"
    },
    {
        "role": "assistant",
        "content": "```sql\n{sql_generated}```"
    }
]
```