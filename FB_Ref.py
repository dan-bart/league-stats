import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import pyarrow.parquet as pq
import pyarrow as pa
import unidecode
import os

class FB_ref_scraper():
    def __init__(self,path,liga):
        self.liga = liga
        self.checkLeagueFolder()
        site = requests.get(path)
        self.main_soup = BeautifulSoup(site.text,"html.parser")


    def checkLeagueFolder(self):
        if not (os.path.isdir('parquet_data/' + self.liga)):
            os.makedirs('parquet_data/' + self.liga)    

            
    def team_sites(self):
        '''
        Returns list of links for every team in specified league
        '''
        table_list = {"laliga":"results107311_overall","ceska_liga":"results107651_overall","italska_liga":"results107301_overall"}
        team_table = self.main_soup.find('table',{'id': table_list[self.liga]})
        tds = team_table.findAll('td', {'data-stat':"squad"})
        self.team_names = [td.text[1:] for td in tds]
        return {td.text[1:]: 'https://fbref.com/' + td.find('a')['href'] for td in tds}

    def team_df(self,team_name,team_site):     
        '''
        Creates a dataframe of every match for given team.
        Extends the dataframe with yellow/red card info and referee
        '''

        comp_list = {"laliga":"La Liga","ceska_liga":"First League","italska_liga":"Serie A"}  
        site = requests.get(team_site)
        soup = BeautifulSoup(site.text,"html.parser")
        match_table = soup.find('table',{'id': "matchlogs_for"})
        df_origin = self.getTeamData(team_name)

        df = pd.read_html(str(match_table))[0]
        df.insert(0,"home_team",team_name)
        df = df.drop(columns = ["Result","Attendance","Captain","Formation","Notes","Match Report"])

        # check if we already have a dataset for this team
        # if so, we only want to load the new data
        if(df_origin is None):
            new_data = df
        else:
            full_df = df_origin
            last_date = df_origin['Date'].max()
            new_data = df.loc[df['Date']>last_date]
        new_data = new_data.loc[new_data['Comp'] == comp_list[self.liga]]

        #extend the dataframe with new data
        if(len(new_data)>0):         
            detail_df = pd.DataFrame(columns = ["match_id","home_yellow_first","home_yellow_second","opponent_yellow_first","opponent_yellow_second"])
            for i,r in new_data.iterrows():
                try:
                    new_row = self.match_details(match_table,r['Date'],team_name)
                    print(new_row)
                    detail_df.loc[i] = new_row
                except:
                    continue

            new_df = pd.concat([new_data, detail_df], axis=1)                 
            if(df_origin is not None):           
                full_df = pd.concat(df_origin,new_df)
            else:
                full_df = new_df
        # standardize and enrich with additional info
        full_df["home_yellow_first_count"] = full_df["home_yellow_first"].apply(len)
        full_df["home_yellow_second_count"] = full_df["home_yellow_second"].apply(len)
        full_df["opponent_yellow_first_count"] = full_df["opponent_yellow_first"].apply(len)
        full_df["opponent_yellow_second_count"] = full_df["opponent_yellow_second"].apply(len)
        full_df["all"] = full_df["home_yellow_first_count"]+full_df["home_yellow_second_count"]+full_df["opponent_yellow_first_count"]+full_df["opponent_yellow_second_count"]
        full_df["first_half"] = full_df["home_yellow_first_count"] + full_df["opponent_yellow_first_count"] 
        full_df["second_half"] = full_df["home_yellow_second_count"] + full_df["opponent_yellow_second_count"]
        full_df = full_df.loc[~full_df["GF"].isnull()]
        full_df = full_df.loc[~full_df["home_yellow_first"].isnull()] 
        full_df = full_df.reset_index()

        if((df_origin is None) or ((df_origin is not None) and len(new_data)>0)):
            self.saveTeamData(team_name,full_df)

        return full_df

    # binomial distribution for a graph
    def card_prob(self,cards):
        dist = dict()
        cards = cards[np.logical_not(np.isnan(cards))]
        for i in np.unique(cards):
            if (bool(dist)):
                 dist[i] = dist[i-1]+ np.count_nonzero(cards == i)/cards.size
            else:
                dist[i] = np.count_nonzero(cards == i)/cards.size
        return dist

    def getTeamData(self,team):
        #remove special characters from team name
        alphanumeric = [character for character in team if character.isalnum()]
        alphanumeric = "".join(alphanumeric)
        correct_name_1 = unidecode.unidecode(alphanumeric)+".parquet"
        print("parquet_data/"+self.liga+"/"+correct_name_1)
        try:
            return pd.read_parquet("parquet_data/"+self.liga+"/"+correct_name_1)
        except:
            return None

    def saveTeamData(self,team,df):
        #remove special characters from team name
        alphanumeric = [character for character in team if character.isalnum()]
        alphanumeric = "".join(alphanumeric)
        correct_name = unidecode.unidecode(alphanumeric)
        #save DF as parquet
        full_path = ".\\parquet_data\\" + self.liga + "\\" + correct_name +".parquet"
        table_pa = pa.Table.from_pandas(df)
        pq.write_table(table_pa, full_path)

    #scrapes deatils of yellow cards given in a match
    def match_details(self,table_soup,date,team_name):
        '''
        Assigns crad info and referee to each match
        '''
        date_str = date.replace('-','')
        cell_with_link = table_soup.find('th',{'csk':date_str})
        detail_link = cell_with_link.find('a')['href'] 
        match_id = detail_link.split("/")[3]
        site = requests.get('https://fbref.com/'+detail_link)
        detail_soup = BeautifulSoup(site.text,"html.parser")
        teams = detail_soup.findAll('div',{'itemprop':'performer'})
        switch_sides = True
        if(teams[0].find('a').text == team_name):
            switch_sides = False

        #get cards
        home_yellow_first = []
        home_yellow_second = []
        opponent_yellow_first = []
        oponnent_yellow_second = []

        match_summary = detail_soup.find('div',{'id':'events_wrap'}).findChild()
        events = match_summary.findAll('div',{'class':["event_header","event a","event b"]})

        first_half = True
        for event in events:
            if(event['class'][0] == "event_header"):
                if event.text == "Kick Off":
                    continue
                if event.text == "Half Time":
                    first_half = False
            else:
                info = event.findAll('div')[2]
                text = info['class'][1]
                if(text != "yellow_card"):
                    continue
                player_name = event.find('a').text

                if(switch_sides):
                    if(first_half):
                        if(event['class'][1] == "b"):
                            home_yellow_first.append(player_name)
                        elif(event['class'][1] == "a"):
                            opponent_yellow_first.append(player_name)
                    if not first_half:
                        if(event['class'][1] == "b"):
                            home_yellow_second.append(player_name)
                        elif(event['class'][1] == "a"):
                            oponnent_yellow_second.append(player_name)
                else:
                     if(first_half):
                        if(event['class'][1] == "a"):
                            home_yellow_first.append(player_name)
                        elif(event['class'][1] == "b"):
                            opponent_yellow_first.append(player_name)
                     if not first_half:
                        if(event['class'][1] == "a"):
                            home_yellow_second.append(player_name)
                        elif(event['class'][1] == "b"):
                            oponnent_yellow_second.append(player_name)                   
                event = event.next_sibling


        return[match_id,home_yellow_first,home_yellow_second,opponent_yellow_first,oponnent_yellow_second]      



    def stats(self,ref, team_1, team_2,main_df):
        '''
        creates a summary statistics of yellow cards given for the teams
        '''
        cards_1_team1 = main_df.loc[main_df["home_team"] == team_1]["home_yellow_first_count"]
        cards_2_team1 = main_df.loc[main_df["home_team"] == team_1]["home_yellow_second_count"]
        self.cards_all_team1 = cards_1_team1+ cards_2_team1

        cards_1_team2 = main_df.loc[main_df["home_team"] == team_2]["home_yellow_first_count"]
        cards_2_team2 = main_df.loc[main_df["home_team"] == team_2]["home_yellow_second_count"]
        self.cards_all_team2 = cards_1_team2+ cards_2_team2

        team_1_all = main_df.loc[main_df["home_team"] == team_1]["all"]
        team_2_all = main_df.loc[main_df["home_team"] == team_2]["all"]
        self.extended = np.append(team_1_all,team_2_all)

        self.cards_1_prob = self.card_prob(self.cards_all_team1)
        self.cards_2_prob = self.card_prob(self.cards_all_team2)
        self.cards_all_prob = self.card_prob(self.extended)

        print("Yellow cards in season 2020/2021 (all referees)")
        print(team_1)
        print(main_df.loc[main_df["home_team"] == team_1][["home_yellow_first_count","home_yellow_second_count"]].agg(["mean","sum"]).round(1))
        print(team_2)
        print(main_df.loc[main_df["home_team"] == team_2][["home_yellow_first_count","home_yellow_second_count"]].agg(["mean","sum"]).round(1))
        print()

        print("Yellow cards in season 2020/2021 (given referee " + ref + ")")
        try:
            print(team_1)
            print(main_df.groupby(by = ["Referee","home_team"]).get_group((ref,team_1))[["home_yellow_first_count","home_yellow_second_count"]].agg(["mean","sum","count"]).round(1))
        except:
            print("Team " + team_1 + " has never played with referee " + ref)
        try:
            print(team_2)
            print(main_df.groupby(by = ["Referee","home_team"]).get_group((ref,team_2))[["home_yellow_first_count","home_yellow_second_count"]].agg(["mean","sum","count"]).round(1))
        except:
            print("Team " + team_2 + " has never played with referee " + ref)
        print()
        print("Chances of getting less than the specified number of cards in the whole game")
        for i in self.cards_all_prob:
            val = round(self.cards_all_prob[i]*100,2)
            print("Less than {}:  {} %".format(i+0.5,val)) 
        print("Chances of getting less than the specified number of cards for team: " + team_1)
        for i in self.cards_1_prob:
            val = round(self.cards_1_prob[i]*100,2)
            print("Less than {}:  {} %".format(i+0.5,val))       
        print("Chances of getting less than the specified number of cards for team: " + team_2)
        for i in self.cards_2_prob:
            val = round(self.cards_2_prob[i]*100,2)
            print("Less than {}:  {} %".format(i+0.5,val)) 
            
    def plot_cdf(self,team = "both"):
        if(team == 1):
            s = pd.Series(self.cards_all_team1, name = 'value')
            df = pd.DataFrame(s)
            stats_df = df.groupby('value')['value'].agg('count').pipe(pd.DataFrame).rename(columns = {'value': 'frequency'})
            # PDF
            stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
            # CDF
            stats_df['cdf'] = stats_df['pdf'].cumsum()
            stats_df = stats_df.reset_index()
            stats_df
            stats_df.plot.bar(x = 'value', y = ['pdf', 'cdf'], grid = True)

        elif(team == 2):
            s = pd.Series(self.cards_all_team2, name = 'value')
            df = pd.DataFrame(s)
            stats_df = df.groupby('value')['value'].agg('count').pipe(pd.DataFrame).rename(columns = {'value': 'frequency'})
            # PDF
            stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
            # CDF
            stats_df['cdf'] = stats_df['pdf'].cumsum()
            stats_df = stats_df.reset_index()
            stats_df
            stats_df.plot.bar(x = 'value', y = ['pdf', 'cdf'], grid = True)
        else:            
            s = pd.Series(self.extended, name = 'value')
            df = pd.DataFrame(s)
            stats_df = df.groupby('value')['value'].agg('count').pipe(pd.DataFrame).rename(columns = {'value': 'frequency'})
            # PDF
            stats_df['pdf'] = stats_df['frequency'] / sum(stats_df['frequency'])
            # CDF
            stats_df['cdf'] = stats_df['pdf'].cumsum()
            stats_df = stats_df.reset_index()
            stats_df
            stats_df.plot.bar(x = 'value', y = ['pdf', 'cdf'], grid = True)
       


def main():
    cesta_sp = "https://fbref.com/en/comps/12/La-Liga-Stats"
    liga_sp = "laliga"

    cesta_it = "https://fbref.com/en/comps/11/Serie-A-Stats"
    liga_it = "italska_liga"

    cesta_cz = "https://fbref.com/en/comps/66/Czech-First-League-Stats"
    liga_cz = "ceska_liga"


    FB = FB_ref_scraper(cesta_it,liga_it)
    sites = FB.team_sites()
    time.sleep(3)
    for key in sites:
        entry = FB.team_df(key,sites[key])
        FB.saveTeamData(key,entry)
        time.sleep(3)
        
if __name__ == "__main__":
    main()