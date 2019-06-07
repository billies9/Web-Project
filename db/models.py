# from FullApp import db
# from . import app
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
# import requests
# import pandas as pd
# import json
# from sqlalchemy import create_engine
# from sqlalchemy.orm import sessionmaker

# db = SQLAlchemy()
metadata = MetaData()
Base = declarative_base(metadata=metadata)

# Models for database tables
class Securities(Base):
    __tablename__ = 'tblSecurities'

    # id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Ticker = Column(String(6), primary_key=True)
    companyName = Column(String(100), nullable=True)
    Price = Column(Float)

# db.create_all()

# Scripts for updating / parsing database tables
def session_builder(engine, function_outputs):
    """ Requirements:
            engine - Required to build connection to database/table
            function-outputs - Formatted as (Table, data)
    """
    Session = sessionmaker(bind = engine)
    s = Session()
    try:
        s.bulk_insert_mappings(function_outputs[0], function_outputs[1])
    except:
        s.rollback()
        s.bulk_update_mappings(function_outputs[0], function_outputs[1])
    s.flush()
    s.commit()
    s.close()
    return

def update_tblSecurities():
    # Initialize connection to database
    engine = create_engine('sqlite:///SecuritiesIndexDB.db', echo=True)

    r_companyName = requests.get('https://financialmodelingprep.com/api/stock/list/all?datatype=json')
    r_realtimePrice = requests.get('https://financialmodelingprep.com/api/v3/stock/real-time-price')

    frame_companyName = pd.DataFrame(json.loads(r_companyName.text))
    frame_realtimePrice = pd.DataFrame(json.loads(r_realtimePrice.text)['stockList'])

    frame_realtimePrice.rename(index=str, columns={'symbol':'Ticker'}, inplace=True)

    merged_frame = pd.merge(frame_companyName, frame_realtimePrice, on='Ticker', how='outer')
    merged_frame.drop('Price', axis=1, inplace=True)

    session_builder(engine, (Securities, merged_frame.to_dict(orient='records')))
    return

# Create new tables if need be without running full app
# def init_db():
#     db.create_all()
#     return



if __name__ == '__main__':
    update_tblSecurities()
