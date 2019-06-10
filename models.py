from sqlalchemy import MetaData, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
import requests
import pandas as pd
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


Base = declarative_base(metadata=MetaData())

# Models for database tables
class Securities(Base):
    __tablename__ = 'tblSecurities'

    # id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    Ticker = Column(String(6), primary_key=True)
    companyName = Column(String(100), nullable=True)
    Price = Column(Float())

    def __repr__(self):
        return '<Securities %r>' % self.companyName

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

def update_tblSecurities(engine):
    r_companyName = requests.get('https://financialmodelingprep.com/api/stock/list/all?datatype=json')
    r_realtimePrice = requests.get('https://financialmodelingprep.com/api/v3/stock/real-time-price')

    frame_companyName = pd.DataFrame(json.loads(r_companyName.text))
    frame_realtimePrice = pd.DataFrame(json.loads(r_realtimePrice.text)['stockList'])

    frame_realtimePrice.rename(index=str, columns={'symbol':'Ticker'}, inplace=True)

    merged_frame = pd.merge(frame_companyName, frame_realtimePrice, on='Ticker', how='inner')
    merged_frame.drop('Price', axis=1, inplace=True)
    merged_frame.rename(index=str, columns={'price':'Price'}, inplace=True)

    # Initialize connection to database and upsert
    session_builder(engine, (Securities, merged_frame.to_dict(orient='records')))
    return

if __name__ == '__main__':
    engine = create_engine('sqlite:///SecuritiesIndexDB.db', echo=True)
    try:
        Base.metadata.create_all(engine)
    except: pass
    update_tblSecurities(engine)
