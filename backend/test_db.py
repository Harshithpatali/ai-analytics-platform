from app.core.database import engine

try:
    connection = engine.connect()

    print(
        "PostgreSQL Connected Successfully!"
    )

    connection.close()

except Exception as error:
    print(
        "Database Connection Failed"
    )

    print(error)