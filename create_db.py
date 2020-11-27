import sqlite3


def create_model_table(model_list):
    c.execute(f"""  CREATE TABLE IF NOT EXISTS models
                    (model_id INTEGER PRIMARY KEY, model VARCHAR);""")
    connection.commit()
    query = f"""INSERT INTO models 
                (model) VALUES"""

    for model in model_list:
        query += f" ('{model}'),"
    query = query.rstrip(',') + ';'

    c.execute(query)
    connection.commit()


def create_nodes_accuracy_table(nodes_list):
    query = """ CREATE TABLE IF NOT EXISTS nodes_accuracy
                (run_id INTEGER PRIMARY KEY, model_id INT,"""

    for nodes in nodes_list:
        query += f' accuracy_{nodes} FLOAT,'
    query = query.rstrip(',') + ');'

    c.execute(query)
    connection.commit()


if __name__ == '__main__':
    connection = sqlite3.connect('accuracy_nodes.db')
    c = connection.cursor()

    models = ['dense', 'eigvector', 'eigvalue', 'sparse']
    create_model_table(models)

    model_nodes = [20, 50] + [100 + 100*n for n in range(20)]
    create_nodes_accuracy_table(model_nodes)

    c.close()
    connection.close()
