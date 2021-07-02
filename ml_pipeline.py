# bare minimum for database and ML model connection
# read from predictions table, write to ml_predictions table
import pymysql
import pandas as pd
from ml_model import ml_arch

if __name__ == '__main__':
    # establish connection to database
    user = 'admin'
    pw = 'kh6hannah'
    host = 'viral-hk-instance.cugmmajsofyv.us-east-1.rds.amazonaws.com'
    db = 'viral_data'
    charset = 'utf8mb4'
    cnx = pymysql.connect(host=host, port=3306, user=user, password=pw, db=db, charset=charset,
                          cursorclass=pymysql.cursors.DictCursor, autocommit=True)

    cur = cnx.cursor()  # cursor: object you use to interact with db
    write = True  # Enables writing back to database

    # -------------------------------------------------------------------------------
    # read predictions table into a dataframe
    get_preds = "SELECT * FROM predictions"
    pred_df = pd.read_sql(get_preds, cnx)
    print(pred_df)

    # -------------------------------------------------------------------------------
    # ML
    ml_df = ml_arch(pred_df)
    print(ml_df)

    # -------------------------------------------------------------------------------
    # write to ml_predictions table
    if write:
        for row in ml_df.itertuples():
            insert_ml = "INSERT INTO `ml_predictions` (`prediction_id`, `ml_new_label`,`ml_intent_label`) " \
                        "VALUES (%s, %s, %s)"
            p_id, ml_il, ml_nl = row.id, row.ml_intent_label, row.ml_new_label
            prediction_id = p_id
            ml_new_label = ml_nl
            ml_intent_label = ml_il
            params = (prediction_id, ml_new_label, ml_intent_label)
            cur.execute(insert_ml, params)

        # Check writing to db was successful
        get_ml = "SELECT * FROM ml_predictions"
        ml_data = pd.read_sql(get_ml, cnx)
        print(ml_data.tail())

    # close db connection
    cur.close()
    cnx.close()
