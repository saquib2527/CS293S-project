with open('234.txt', 'r') as fh:
    contents = fh.readlines()
    for x in range(0, len(contents), 2):
        current = contents[x].strip()
        next = contents[x + 1].strip()
        station_id, num_neighbors, temp, mse = current.split(',')
        mse = float(mse)
        mse_local = float(next.split(',')[3])
        print('{0} & {1} & {2} & {3}\\\\\n\\hline'.format(station_id, num_neighbors, format(mse, '.8f'), format(mse_local, '.8f')))
