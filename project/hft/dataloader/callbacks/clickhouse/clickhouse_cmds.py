create_snapshots = 'CREATE TABLE IF NOT EXISTS snapshots (moment DateTime, millis UInt16, symbol FixedString(10), \
                   x0 Float32,x1 UInt32,x2 Float32,x3 UInt32,x4 Float32,x5 UInt32,x6 Float32,x7 UInt32,x8 Float32,\
                   x9 UInt32,x10 Float32,x11 UInt32,x12 Float32,x13 UInt32,x14 Float32,x15 UInt32,x16 Float32,\
                   x17 UInt32,x18 Float32,x19 UInt32,x20 Float32,x21 UInt32,x22 Float32,x23 UInt32,x24 Float32,\
                   x25 UInt32,x26 Float32,x27 UInt32,x28 Float32,x29 UInt32,x30 Float32,x31 UInt32,x32 Float32,\
                   x33 UInt32,x34 Float32,x35 UInt32,x36 Float32,x37 UInt32,x38 Float32,x39 UInt32,x40 Float32,\
                   x41 UInt32,x42 Float32,x43 UInt32,x44 Float32,x45 UInt32,x46 Float32,x47 UInt32,x48 Float32,\
                   x49 UInt32,x50 Float32,x51 UInt32,x52 Float32,x53 UInt32,x54 Float32,x55 UInt32,x56 Float32,\
                   x57 UInt32,x58 Float32,x59 UInt32,x60 Float32,x61 UInt32,x62 Float32,x63 UInt32,x64 Float32,\
                   x65 UInt32,x66 Float32,x67 UInt32,x68 Float32,x69 UInt32,x70 Float32,x71 UInt32,x72 Float32,\
                   x73 UInt32,x74 Float32,x75 UInt32,x76 Float32,x77 UInt32,x78 Float32,x79 UInt32,x80 Float32,\
                   x81 UInt32,x82 Float32,x83 UInt32,x84 Float32,x85 UInt32,x86 Float32,x87 UInt32,x88 Float32,\
                   x89 UInt32,x90 Float32,x91 UInt32,x92 Float32,x93 UInt32,x94 Float32,x95 UInt32,x96 Float32,\
                   x97 UInt32,x98 Float32,x99 UInt32) \
                   ENGINE=File(CSV) '

create_orderbook10 = 'CREATE TABLE IF NOT EXISTS orderbook_10_03_20 (moment DateTime, millis UInt16, symbol FixedString(10), \
                      a1 Float32, a2 Float32, a3 Float32, a4 Float32, a5 Float32, a6 Float32, a7 Float32, a8 Float32, a9 Float32, a10 Float32, \
                      av1 UInt32, av2 UInt32, av3 UInt32, av4 UInt32, av5 UInt32, av6 UInt32, av7 UInt32, av8 UInt32, av9 UInt32, av10 UInt32, \
                      b1 Float32, b2 Float32, b3 Float32, b4 Float32, b5 Float32, b6 Float32, b7 Float32, b8 Float32, b9 Float32, b10 Float32, \
                      bv1 UInt32, bv2 UInt32, bv3 UInt32, bv4 UInt32, bv5 UInt32, bv6 UInt32, bv7 UInt32, bv8 UInt32, bv9 UInt32, bv10 UInt32) \
                      ENGINE=File(CSV) '

create_trades = 'CREATE TABLE IF NOT EXISTS trades_orderbook_10_03_20 (symbol FixedString(15), moment DateTime, millis UInt16, price Float32, size UInt32, \
      action FixedString(15), side FixedString(5)) \
      ENGINE=File(CSV)'

create_indexes = 'CREATE TABLE IF NOT EXISTS indexes_10_03_20 (symbol FixedString(15), moment DateTime, price Float32) \
                  ENGINE=File(CSV)'