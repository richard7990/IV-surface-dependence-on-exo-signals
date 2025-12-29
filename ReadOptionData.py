def make_connection():
    """
    Make a connection to the database
    :return: connection object
    """
    # make the connection
    conn = mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="",  # confirm this is correct for your server
        database="options",
        port=3306,
        connection_timeout=5,
        auth_plugin="mysql_native_password",
        use_pure=True)
    return conn

def get_option_data(ticker, start_date = None, end_date = None):
    """
    Build the sql query and returns a dataframe of the option data for a given ticker
    :param ticker: ticker symbol
    :param start_date: datetime format start date
    :param end_date: datetime format end date
    :return: df of ticker
    """
    if start_date is None:
        # reach as far back as possible
        start_date = '1900-01-01'
    if end_date is None:
        end_date = date.today().strftime('%Y-%m-%d')

    # build the query
    QUERY = """
    SELECT *
    FROM option_chain
    WHERE act_symbol = $my_symbol$ AND date BETWEEN $my_start$ AND $my_end$
    """
    QUERY = QUERY.replace('$my_symbol$', ticker)
    QUERY = QUERY.replace('$my_start$', start_date)
    QUERY = QUERY.replace('$my_end$', end_date)

    # make the connection
    conn = make_connection()

    if conn.is_connected():
        df = pd.read_sql(QUERY, conn)
        conn.close()
    return df



