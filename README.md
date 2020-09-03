# This Is Happening

Locate pockets of social activity in meatspace and tweet about it.

This app detects geotagged Twitter activity that stands out above typical background levels.

## TODO

### Infrastructure

- [ ] Use Redis for historical stats. Key: year, month, day, hour. Will be able to store more historical data (e.g., 1 week).

### ML for event finding

- [ ] When an event is found, run DBSCAN on all recent tweets across all tiles to determine the full set of event tweets, some of which may fall outside of the current tile. Need to ensure the resulting cluster includes tweets from the original event detection.

### Data

- [ ] Detect and ignore spam tweets, e.g., job postings, apartment listings

### Analytics

- [ ] Plot the pulse of a neighborhood over time

### Maps

- [ ] Setup: Plot preview of proposed map tiles
  - [ ] Estimate tile area
- [ ] When an event is found, tweet an image of a map with the location/heat map
