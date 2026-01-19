# Financial Data Pipeline - Technical Documentation

## System Architecture

This pipeline implements a strict, reproducible data processing workflow for financial market data.

### Design Principles

1. **Immutability**: Raw data is never modified; all transformations create new datasets
2. **Causality**: No look-ahead bias; features use only past data
3. **Reproducibility**: Pipeline produces identical results given same inputs
4. **Traceability**: All validation issues are logged with timestamps and reasons
5. **Modularity**: Each stage can run independently

---

## Data Flow

```
Yahoo Finance API
       ↓
[Raw OHLCV + Adjusted Prices]
       ↓
data/raw/{timeframe}/{symbol}_{timeframe}_raw.csv
       ↓
[Validation Checks]
       ↓
data/validated/{timeframe}/{symbol}_{timeframe}_validated.csv (with flags)
data/validated/{timeframe}/clean/{symbol}_{timeframe}_clean.csv (clean only)
data/validated/validation_log.csv (issues log)
       ↓
[Feature Engineering]
       ↓
data/features/{timeframe}/{symbol}_{timeframe}_features.csv
```

---

## Module Reference

### 1. Configuration (`src/config/settings.py`)

Central configuration for all pipeline parameters.

**Key Settings:**
- `DEFAULT_EQUITY_SYMBOLS`: List of equity tickers
- `DEFAULT_INDEX_SYMBOLS`: List of index tickers
- `TIMEFRAMES`: Supported intervals (1m, 5m, 1h, 1d)
- `DEFAULT_LOOKBACK_DAYS`: Historical data fetch periods per timeframe
- `VALIDATION_CONFIG`: Thresholds for data quality checks
- `FEATURE_CONFIG`: Parameters for technical indicators

### 2. Data Ingestion (`src/ingestion/`)

**`equity_ohlcv.py`** - Core ingestion logic
- `YahooFinanceIngestion`: Main class for data fetching
  - `fetch_ohlcv()`: Get raw OHLCV data
  - `fetch_adjusted_prices()`: Get split/dividend-adjusted data
  - `merge_raw_and_adjusted()`: Combine both datasets
  - `save_data()`: Persist to CSV
  - `load_data()`: Read from CSV

**`runner.py`** - Ingestion orchestrator
- `run_ingestion()`: Execute complete ingestion pipeline

**Output Columns:**
```
timestamp, symbol, open, high, low, close, volume,
adj_open, adj_high, adj_low, adj_close, adj_volume,
dividends, stock splits
```

### 3. Data Validation (`src/validation/`)

**`ohlcv_checks.py`** - Validation logic
- `OHLCVValidator`: Main validation class
  - `validate_price_relationships()`: Check high≥low, etc.
  - `validate_volume()`: Ensure volume≥0
  - `detect_price_outliers()`: Flag abnormal price changes
  - `detect_duplicates()`: Find duplicate timestamps
  - `detect_missing_timestamps()`: Identify gaps
  - `validate_dataset()`: Run all checks
  - `get_clean_dataset()`: Extract valid records

**`validation_runner.py`** - Validation orchestrator
- `run_validation()`: Execute complete validation pipeline

**Validation Flags Added:**
```
valid_price_relationship, valid_volume, is_outlier,
is_duplicate, has_gap, is_valid
```

### 4. Feature Engineering (`src/features/`)

**`technical_indicators.py`** - Feature computation
- `TechnicalIndicators`: Main indicator class

**Trend Indicators:**
- `sma()`: Simple Moving Average
- `ema()`: Exponential Moving Average
- `macd()`: MACD indicator with signal and histogram
- `adx()`: Average Directional Index

**Momentum Indicators:**
- `rsi()`: Relative Strength Index
- `roc()`: Rate of Change
- `stochastic_oscillator()`: Stochastic %K and %D

**Volatility Indicators:**
- `atr()`: Average True Range
- `bollinger_bands()`: Bollinger Bands with width
- `historical_volatility()`: Annualized volatility

**Volume Indicators:**
- `vwap()`: Volume Weighted Average Price
- `obv()`: On-Balance Volume
- `volume_roc()`: Volume Rate of Change

**`feature_runner.py`** - Feature orchestrator
- `run_feature_engineering()`: Execute complete feature pipeline

---

## Feature Specifications

### Trend Features

| Feature | Formula | Periods | Notes |
|---------|---------|---------|-------|
| SMA | Mean(Close, N) | 10, 20, 50, 200 | Simple moving average |
| EMA | EMA(Close, N) | 12, 26 | Exponential moving average |
| MACD | EMA(12) - EMA(26) | 12, 26, 9 | With signal line and histogram |
| ADX | Complex | 14 | Trend strength indicator |

### Momentum Features

| Feature | Formula | Periods | Range |
|---------|---------|---------|-------|
| RSI | 100 - (100/(1+RS)) | 14 | 0-100 |
| ROC | ((Close - Close_N) / Close_N) * 100 | 5, 10, 20 | % |
| Stoch %K | ((Close - Low_N) / (High_N - Low_N)) * 100 | 14 | 0-100 |
| Stoch %D | SMA(Stoch %K, 3) | 3 | 0-100 |

### Volatility Features

| Feature | Formula | Periods | Notes |
|---------|---------|---------|-------|
| ATR | Mean(True Range, N) | 14 | Average True Range |
| BB Upper | SMA(N) + (2 * StdDev) | 20 | Bollinger upper band |
| BB Lower | SMA(N) - (2 * StdDev) | 20 | Bollinger lower band |
| BB Width | BB Upper - BB Lower | 20 | Band width |
| Hist Vol | StdDev(Returns, N) * √252 | 10, 20, 60 | Annualized |

### Volume Features

| Feature | Formula | Notes |
|---------|---------|-------|
| VWAP | Σ(Price * Volume) / Σ(Volume) | Cumulative per day |
| OBV | Cumulative volume direction | Based on close direction |
| Vol ROC | ((Vol - Vol_N) / Vol_N) * 100 | Periods: 5, 10, 20 |

---

## Data Quality Checks

### 1. Price Relationship Validation
- `high >= low` (always)
- `high >= close` (always)
- `high >= open` (always)
- `low <= close` (always)
- `low <= open` (always)

### 2. Volume Validation
- `volume >= 0` (always)

### 3. Outlier Detection
- Flag if `|close - open| / open > 20%`
- Configurable threshold in settings

### 4. Duplicate Detection
- Check for multiple records with same timestamp
- Keep first occurrence in clean data

### 5. Missing Timestamp Detection
- Expected frequency based on timeframe:
  - 1m: 1 minute intervals
  - 5m: 5 minute intervals
  - 1h: 1 hour intervals
  - 1d: 1 day intervals
- Flag gaps > 2x expected interval (allows for weekends/holidays)

---

## NaN Handling

### Expected NaN Values

Technical indicators require historical data to compute. Early records will have NaN values:

| Indicator | NaN Period | Reason |
|-----------|------------|--------|
| SMA(10) | First 9 records | Needs 10 periods |
| SMA(200) | First 199 records | Needs 200 periods |
| RSI(14) | First 13 records | Needs 14 periods |
| MACD Signal | First 33 records | Needs MACD + 9 periods |
| ATR(14) | First 13 records | Needs 14 periods |

**Best Practice:** Drop first 200 rows when using daily data with all indicators.

---

## Performance Considerations

### Data Volume

For the default configuration:
- **1m timeframe**: ~1,560 records/symbol (7 days)
- **5m timeframe**: ~2,890 records/symbol (60 days)
- **1h timeframe**: ~3,476 records/symbol (2 years)
- **1d timeframe**: ~250 records/symbol (1 year)

### Execution Time

On typical hardware:
- Ingestion: ~10-15 seconds (API dependent)
- Validation: ~2-3 seconds
- Feature Engineering: ~3-5 seconds
- **Total**: ~15-20 seconds for full pipeline

### Memory Usage

- Raw data in memory: ~10-20 MB
- With features: ~30-50 MB
- CSV files total: ~50-100 MB

---

## Error Handling

### Network Errors
- Individual symbol fetch failures are logged but don't stop pipeline
- Retry logic can be added to `YahooFinanceIngestion`

### Data Validation Failures
- Invalid records are flagged but not removed from validated dataset
- Clean dataset contains only valid records
- All issues logged to `validation_log.csv`

### Feature Computation Errors
- NaN values expected for early periods
- Division by zero handled gracefully (returns NaN)
- Invalid indicators logged but don't crash pipeline

---

## Extending the Pipeline

### Adding New Data Sources

1. Create new ingestion class in `src/ingestion/`
2. Implement same interface as `YahooFinanceIngestion`
3. Update `runner.py` to call new source
4. Ensure UTC timestamp normalization

### Adding New Validation Checks

1. Add method to `OHLCVValidator` class
2. Call in `validate_dataset()` method
3. Update validation log structure if needed

### Adding New Indicators

1. Add method to `TechnicalIndicators` class
2. Call in `generate_all_features()` method
3. Add configuration to `FEATURE_CONFIG` in settings
4. Document expected NaN period

### Adding New Timeframes

1. Add to `TIMEFRAMES` list in settings
2. Add lookback period to `DEFAULT_LOOKBACK_DAYS`
3. Update frequency mapping in validation

---

## Testing

### Unit Tests (Recommended)

```python
# Test data ingestion
def test_fetch_ohlcv():
    ingestion = YahooFinanceIngestion()
    data = ingestion.fetch_ohlcv(
        ['AAPL'],
        start_date=datetime(2025, 1, 1),
        end_date=datetime(2025, 1, 31),
        interval='1d'
    )
    assert 'AAPL' in data
    assert len(data['AAPL']) > 0

# Test validation
def test_price_validation():
    validator = OHLCVValidator()
    df = pd.DataFrame({
        'timestamp': [datetime.now()],
        'symbol': ['TEST'],
        'high': [100],
        'low': [95],
        'open': [98],
        'close': [99],
        'volume': [1000]
    })
    validated, log = validator.validate_dataset(df, '1d')
    assert validated['is_valid'].all()
```

### Integration Tests

Run complete pipeline on test data and verify outputs exist.

---

## Common Issues & Solutions

### Issue: No data returned for symbol
**Solution:** Check if symbol is valid on Yahoo Finance; some symbols require special formatting

### Issue: Timestamp gaps detected
**Solution:** Normal for weekends/holidays; adjust `max_gap_multiplier` if too sensitive

### Issue: Too many NaN values in features
**Solution:** Expected for early periods; filter or use shorter lookback periods

### Issue: Price outliers detected
**Solution:** Verify if genuine (e.g., earnings, splits); adjust threshold if needed

### Issue: Memory error with large datasets
**Solution:** Process fewer symbols at once or reduce lookback periods

---

## Best Practices

1. **Always validate before feature engineering**
2. **Keep raw data immutable** - never modify original CSV files
3. **Monitor validation logs** - review issues regularly
4. **Handle NaN appropriately** - drop or forward-fill based on use case
5. **Version your data** - timestamp output directories
6. **Test on small datasets first** - verify before scaling up
7. **Set appropriate lookback periods** - balance history vs. API limits
8. **Use adjusted prices for backtesting** - accounts for corporate actions

---

## API Limits & Constraints

### Yahoo Finance Limitations

- **1m data**: Limited to 7 days
- **5m data**: Limited to 60 days  
- **1h data**: Available for ~2 years
- **1d data**: Available for decades

- **Rate limiting**: Possible with high-frequency requests
- **Market hours**: Intraday data only during trading hours
- **Delisted stocks**: May have incomplete data

**Recommendation:** Add delays between requests if fetching many symbols.

---

## Maintenance

### Regular Tasks

1. **Update dependencies**: `pip install --upgrade -r requirements.txt`
2. **Verify data quality**: Check validation logs weekly
3. **Monitor disk usage**: Clean old CSV files if needed
4. **Backup configurations**: Version control settings.py changes

### Monitoring

Track these metrics:
- Data fetch success rate
- Validation failure rate
- Pipeline execution time
- Disk space usage

---

## License & Disclaimer

**Educational Use Only**

This pipeline is for educational and research purposes. Market data accuracy depends on Yahoo Finance API reliability. Always verify critical data from primary sources.

**Not Financial Advice**

This tool is for data processing only. It does not provide trading signals, investment advice, or recommendations.

---

## Support & Contributing

For questions or improvements:
1. Review this documentation first
2. Check the README.md for quick reference
3. Review examples.py for usage patterns
4. Extend the pipeline following the patterns shown

**Code Quality Standards:**
- Follow PEP 8 style guidelines
- Add type hints where possible
- Document all public methods
- Log important operations
- Handle errors gracefully
