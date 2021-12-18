

ASSETS = ['Binance Coin',
          'Bitcoin',
          'Bitcoin Cash',
          'Cardano',
          'Dogecoin',
          'EOS.IO',
          'Ethereum',
          'Ethereum Classic',
          'IOTA',
          'Litecoin',
          'Maker',
          'Monero',
          'Stellar',
          'TRON']

ID2ASSET = {i: asset for i, asset in enumerate(ASSETS)}
ASSET2ID = {asset: i for i, asset in enumerate(ASSETS)}

DATA_PERIODS = ['2021',
                '2020',
                '2019',
                '2018',
                '2017',
                'COMP',
                'CUPP']