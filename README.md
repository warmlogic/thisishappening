# This Is Happening

Locate, summarize, and visualize pockets of social activity in meatspace.

This app detects geotagged Twitter activity that stands out above typical background levels and tweets about it.

Example account: [https://twitter.com/happening_sf](https://twitter.com/happening_sf)

## Run the application

### As a Heroku app

These instructions use the Heroku CLI

1. Fork this repo on GitHub and ensure you have a branch called `main`
1. Create a new app on Heroku: `heroku create my-app-name`
1. Install add-ons for:
   1. [Papertrail](https://elements.heroku.com/addons/papertrail)
      1. `heroku addons:create papertrail -a my-app-name`
   1. [Postgres](https://elements.heroku.com/addons/heroku-postgresql)
      1. `heroku addons:create heroku-postgres -a my-app-name`
1. Create a new token: `heroku authorizations:create -d "my cool token description"`
   1. Add the token to your GitHub repo's Secrets under the name `HEROKU_API_KEY`
1. Add your Heroku app's name to the GitHub repo's Secrets under the name `HEROKU_APP_NAME` (or however it is configured in `.github/workflows/deploy.yaml`)
1. Configure the application by adding environment variables as [Config Vars](https://devcenter.heroku.com/articles/config-vars)
1. Commit and push to your GitHub repo's `main` branch
   1. This can be through committing a change, merging a PR, or just running `git commit -m "empty commit" --allow-empty`
   1. This will use GitHub Actions to build the app using Docker and deploy to Heroku

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
- [x] Exclude my own tweets from the search (put myself in the list of users to ignore)
- [ ] Plot the pulse of a neighborhood over time: count of tweets by hour

## Realizations

- Many tweets show up at the canonical city location, especially due to Swarm's "posted a photo" feature
- Using tiles / artificial region boundaries required more workarounds and convoluted solutions than originally expected. Some downsides:
  - Potentially splits events across regions
  - Keeping the running statistics requires storing many rows in database table; wouldn't be an issue if I wasn't trying to operate on a shoestring budget because I could run my own database
- It's not uncommon to get a false alarm due to one user posting many tweets in a short time period
- To add a location to the bot tweets, need to enable in: Settings and privacy -> Privacy and safety -> Location information -> Add location information to your Tweets
