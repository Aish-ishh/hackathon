#automatically block spam in whatsapp using java?
import requests
import json
import os
import time
!pip install validators
import validators
def main():
    try:
        def apicheck():
                try:
                    api = input("please enter your virus total api key: ")
                    url = 'https://www.virustotal.com/vtapi/v2/url/report'
                    params = {'apikey': api, 'url': "https://google.com" }
                    requests.get(url, params=params)
                    return api
                except Exception as e:
                    print("that api code does not seem to be valid please try again")
                    apicheck()     
        print("valid api code")
        def pickurlandchecking():
            try:
                inputforurl = input("enter a url to scan: ")







