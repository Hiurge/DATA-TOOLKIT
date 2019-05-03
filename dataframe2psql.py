import psycopg2
import psycopg2.extras
import pandas as pd


# Move dataframe to psql as a table
def df2psql_table(df, credencials):

	# Setup psql table schema out of df
	table_parts = ['(id serial PRIMARY KEY']
	for column_name in df.columns:
		if   df[column_name].dtype == 'object':  dtype = 'text'
		elif df[column_name].dtype == 'int64':   dtype = 'integer'
		elif df[column_name].dtype == 'float64': dtype = 'float'
		table_parts.append( '"{}" {}'.format(column_name, dtype))
	table_schema = ', '.join(table_parts) + ');'

	# df values into psql insert format
	columns = ','.join(['"{}"'.format(column) for column in list(df)])
	values = 'VALUES({})'.format(','.join(["%s" for _ in list(df)]))
	insert = 'INSERT INTO {} ({}) {}'.format(credencials['table_name'], columns, values)
	
	# PSQL connection
	conn = psycopg2.connect('dbname={} user={}'.format(credencials['dbname'], credencials['user']))
	cur = conn.cursor()
	
	# Load schema
	cur.execute('CREATE TABLE {} {}'.format(credencials['table_name'], table_schema))

	# Load values into schema
	psycopg2.extras.execute_batch(cur, insert, df.values)

	conn.commit()
	conn.close()
	cur.close()


# Example:
#df = pd.read_csv('your.csv') # or # df = your_dataframe
#credencials = {'dbname':'your_db_name', 'user':'your_db_user_name', 'table_name':'name_your_stuff_storage'}
#df2psql_table(df, credencials)

# To expand db use of credencials (fg. password), edit line 24.
