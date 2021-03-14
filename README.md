# This Is Happening

Locate, summarize, and visualize pockets of social activity in meatspace.

This app detects geotagged Twitter activity that stands out above typical background levels and tweets about it.

Example account: [https://twitter.com/happening_sf](https://twitter.com/happening_sf)

## TODO

### Data

- Tweets
  - [x] Use a density-based metric for event detection, rather than aggregating within pre-defined boundaries (map tiles) and tracking statistics for each tile
    - [ ] Define activity thresholds using intuitive, human readable values
  - [x] Prevent a single prolific user from easily triggering an event by decreasing the weight of their tweets
  - [x] Deduplicate tokens within each tweet
  - [x] Reduce weight for tweets with specific longitude and latitude (e.g., "canonical" city locations that get assigned to Instagram photo posts)
  - [ ] Detect and ignore spam tweets, e.g., job postings, apartment listings
- Queries
  - [x] Provide access to the tweets associated with each event
  - [x] Write query to keep most recent N days of data and run in main loop
  - [x] Maintain maximum recent_tweets table row count and run in main loop

### ML for event finding

- [x] Clustering
  - [x] When an event is found, run a clustering algorithm (DBSCAN) on all recent tweets to determine the full set of event tweets
  - [x] Define cluster neighborhood limits using intuitive, human readable values (e.g., kilometers)

### Publishing / Analytics

- [x] When an event is found, tweet an image of a map with the location/heat map
- [x] Set my tweet's location to the event latitude and longitude
- [ ] Exclude my own tweets from the search
- [ ] Plot the pulse of a neighborhood over time: count of tweets by hour

## Realizations

- Many tweets show up at the canonical city location, especially due to Swarm's "posted a photo" feature
- Using tiles / artificial region boundaries required more workarounds and convoluted solutions than originally expected. Some downsides:
  - Potentially splits events across regions
  - Keeping the running statistics requires storing many rows in database table; wouldn't be an issue if I wasn't trying to operate on a shoestring budget because I could run my own database
- It's not uncommon to get a false alarm due to one user posting many tweets in a short time period
- To add a location to the bot tweets, need to enable in: Settings and privacy -> Privacy and safety -> Location information -> Add location information to your Tweets
