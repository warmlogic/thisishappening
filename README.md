# This Is Happening

Locate, summarize, and visualize pockets of social activity in meatspace.

This app detects geotagged Twitter activity that stands out above typical background levels and tweets about it.

Example account: [https://twitter.com/happening_sf](https://twitter.com/happening_sf)

## TODO

### ML for event finding

- [ ] Clustering
  - [x] Cluster tweets when event threshold exceeded
  - [x] When an event is found, run DBSCAN on all recent tweets across all tiles to determine the full set of event tweets, since some tweets may fall outside of the current tile
  - [ ] Ensure one of the resulting clusters includes tweets from the original event detection (by tile id)

### Data

- [ ] Separate "canonical city location" tweets from actually geotagged activity using a specific longitude+latitude.
- [ ] Detect and ignore spam tweets, e.g., job postings, apartment listings
- [x] Provide access to the tweets associated with each event
- [x] Write query to keep most recent N days of data and run in main loop
- [x] Maintain maximum recent_tweets table row count and run in main loop
- [x] Deduplicate tokens within each tweet

### Analytics

- [ ] Plot the pulse of a neighborhood over time: count of tweets by hour

### Maps

- [x] When an event is found, tweet an image of a map with the location/heat map

## Realizations

- Many tweets show up at the canonical city location, especially due to Swarm's "posted a photo" feature
- Using tiles / artificial region boundaries has lead to more difficulties and workarounds than originally expected
  - Potentially splits events across regions
  - Keeping the running statistics requires storing many rows in database table; wouldn't be an issue if I wasn't trying to operate on a shoestring budget because I could run my own database
- Not uncommon to get a false alarm due to one user posting many tweets in a short time period
