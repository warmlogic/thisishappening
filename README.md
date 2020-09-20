# This Is Happening

Locate, summarize, and visualize pockets of social activity in meatspace.

This app detects geotagged Twitter activity that stands out above typical background levels and tweets about it.

Example account: [https://twitter.com/happening_sf](https://twitter.com/happening_sf)

## TODO

### Infrastructure

- [ ] Use Redis for historical stats. Key: year, month, day, hour. Will be able to store more historical data (e.g., 1 week).

### ML for event finding

- [ ] Cluster tweets when event threshold exceeded
  - [x] When an event is found, run DBSCAN on all recent tweets across all tiles to determine the full set of event tweets, since some tweets may fall outside of the current tile
  - [ ] Ensure one of the resulting clusters includes tweets from the original event detection (by tile id)

### Data

- [ ] Detect and ignore spam tweets, e.g., job postings, apartment listings
- [x] Provide access to the tweets associated with each event
- [ ] Maintain maximum recent_tweets table row count
  - [x] Write query to keep N rows for
  - [ ] Run query in main loop
- [ ] Separate "canonical city location" tweets from actually geotagged activity using a specific longitude+latitude.

### Analytics

- [ ] Plot the pulse of a neighborhood over time: count of tweets by hour

### Maps

- [ ] Setup: Plot preview of proposed map tiles (use folium)
  - [ ] Estimate tile area
- [x] When an event is found, tweet an image of a map with the location/heat map
