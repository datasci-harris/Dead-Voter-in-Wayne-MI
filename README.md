# Dead-Voter-in-Wayne-MI

The original idea of this project is to examine the dataset (DeadVoter.txt) posted on Twitter about the dead voters in Michigan by pushing these records to the public Michigan voter file data website mvic.sos.state.mi.us/Voter/Index and take out all the suspicious data (voters whose age is over 100) to see what is happening.

After this, we will then scrap more Michigan counties' death record from MyHeritage or other similar websites and use these death record to cross check the voter info to see whether there are real 'dead voters'. If we could find any 'dead voters' that could not be simply explained by information deficiency or manual typing mistake, we will then try to merge more data regarding incomes, political preference, and so on, to do some data analysis and machine learning. The core goal of all these work will be to know whether there is some distribution difference of 'dead voters' among different counties and if so, ***whether they are related to the 2020 General Election result***.

However, the actual implementation varied a little bit from our original design, but we believe, we still got what we want to know. More details will be explained below.


## Part 1 Data Collection
The original second stage design of this project was planned to refer to the ZombieVoters project [https://github.com/BenWirus/ZombieVoters.git] developed by BenWirus. However, because of the verification of the OAuth token keeps running into failure, I failed at scrapping from MyHeritage. And since the due date is coming, to continue to work on this becomes a little bit risky. Ziyi and I decided to go on our analysis with the dataset we obtained from the first stage scrapping and some other datasets I obtained from the Accountability Project. We would like to come back to this during the winter break to see what is happening in a broader area.

The good news is, although we are unable to ensure 'dead voters', using the 'over-100' standard in scrapper.py, we could still partially observe whether the existence of these doubtable voters has something do do with the vote result. And we get a bigger voter file of Wayne County in MI. ***Wayne county itself could be divided into two parts, Detroit city and other cities, which has delivered totally different result shown in the General Election Counting work.*** So we could use these two areas to resemble counties that have different election results. The logic is not fully scientific, but the code will work perfectly through, which makes it reusable when we finished the second stage scrapping work in the future.

And here are the detailed introduction on the other datasets that we used in this project:
1. The unexamined dead voter dataset comes from the rumor twitter which has already been blocked.
2. The registered voter dataset comes from the Accountability Project. I contacted their admin analyst, Jennifer LaFleur, and get this dataset. It is not a whole voter registration file. And Jennifer told me that they got these file through the Wayne county information office.
3. These three files in out folder were scrapped from the voter registration check page of MI [https://mvic.sos.state.mi.us/Voter/SearchByName] using the scrapper.py that I created before. MI government has closed this page after I got these data, so this py file could not be straightforward used now, but they still have another page [https://mvic.sos.state.mi.us/Voter] which was designed using the same frame, so I believe it will still work with some minor change, but since the due date of the final project is coming, I will not spend too much time to update this now, maybe in the winter holiday.
4. The income file comes from the Internal Revenue Service, https://www.irs.gov/statistics/soi-tax-stats-individual-income-tax-statistics-2018-zip-code-data-soi
5. The zipcode file was simply got from Zip-codes.com, https://www.zip-codes.com/county/mi-wayne.asp


## Part 2 Data Analysis
### Plotting
Since the Wayne registered voter dataset is not complete, and the over-100 registered voter dataset has no race attribute, this race distribution we plot out here [plot/Registered Voters by Race in Wayne County (Incomplete Data).png] is not complete. But it seems like all the over-100 registered voters are either white or African, which is a little bit weird.

It becomes even stranger when we come to the distribution of the over-100 voters' age. We have done the crosscheck work for the over-100 voter dataset. The methodology behind is to identify the over-100 voters with the spreading fraud voter dataset. But the problem is, the age of these people seems perfectly normally distributed, which is super uncommon.

### Testing
We mainly did two t-test and made a linear regression model here. The first t-test is aimed at testing the difference of over-100 vote stuats distribution between Detroit and non-Detroit area. And the second t-test is used to test the difference of age distribution between Detroit and non-Detroit area in Wayne County. Apparently, both of them told us to reject the null hypothesis.

The linear regression model here is a simplified one. If we could access to more data related to zipcode, we could probably increase its explanation power. Anyway, the current model tell us, when controlling the income, the percentage of the 'over-100' voters could explain a big percentage of the possibility of such a region being supportive of Democrats.

### Machine Learning
We then tried to make some machine learning model. We tested several supervised models, although most of them do not predict very well. Surprisingly, both dimensionality reduction and clustering delivered really beautiful result. So we believe, if we could have more county-level or zipcode-level data to better train this model, we might be able to predict the result using the percentage of the over-100 voter.

### GIS mapping
We also tried to map the amount of over-100 voters and some other variables onto the map of Wayne County area. You could see this part of work in the Jupyter notebook.


## Part 3 Finding and thoughts
Overall, based on the modeling, data analysis work, and our qualitative analysis, ***we believe the distribution of the over-100 voters in pro-Democrats area is pretty doubtable***. This may not lead to a conclusion of whether there exist fraud voting behavior or not, but I do believe we need to invest more time and energy to obtain more data and further follow this topic.
