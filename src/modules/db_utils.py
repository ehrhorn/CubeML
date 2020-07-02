import numpy as np
from src.modules.helper_functions import get_time

def create_transformed_db(old_db, new_db, transformers, chunksize=100000):
    # Expects new_db, which is a copy of the old db.
    # If a key from old_db is in transformers --> 
    # load --> transofrm --> save in new

    db_tables = old_db.tables
    tables = ['sequential', 'scalar', 'meta']
    primary_keys = ['row', 'event_no', 'event_no']

    # Since different DBs start at different indices, find the beginning
    id0 = int(old_db.ids[0])
    
    for table, primary_key in zip(tables, primary_keys):
        for var in db_tables[table]:
            if var in transformers:
                print('')
                if var in [
                    'dom_x',
                    'dom_y',
                    'dom_z',
                    'dom_charge',
                    'dom_time',
                    'dom_pulse_width'
                ]:
                    continue
                print(get_time(), 'Transforming', var, '...')
                # We found avariable that needs transforming. Transform it!
                transformer = transformers[var]

                # Loop over the primary key in the table
                i = 0
                if primary_key == 'row':
                    start = 0
                else:
                    start = id0

                # Keep transforming until all have been transformed
                while True:
                    _from = start + i*chunksize 
                    _to = start + (i+1)*chunksize
                    indices = [
                        str(e) for e in np.arange(_from, _to)
                        ]
                    print(get_time(), 'Transforming %s - %s'%(indices[0], indices[-1]))
                    # print(indices[:10], old_db.ids[:10])
                    fetched = old_db.read(table, var, primary_key, indices)
                    n_fetched = len(fetched)
                    transformed = np.squeeze(
                        transformer.transform(
                            fetched.reshape(-1, 1)
                        )
                    )

                    # Write to new db
                    new_db.write(
                        table, var, 
                        indices[:n_fetched], 
                        transformed, 
                        primary_key=primary_key
                    )
                    # Check if we reached the end
                    if n_fetched < chunksize:
                        print(get_time(), 'Transformation of %s finished.'%(var))
                        break
                    else:
                        i += 1
                    


