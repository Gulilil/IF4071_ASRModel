import pandas as pd

LABELS = '''
Call an ambulance now!
There's a fire in the building.
Someone is hurt; we need help!
Dial 911 immediately.
Police, there's a robbery in progress.
We need a firefighter here quickly.
I see smoke coming from that house.
Someone is trapped in the elevator.
The car crash needs an ambulance.
Call the police for assistance.
There's an intruder in my home.
The child is missing; call the police.
Help! Someone is drowning.
The fire alarm is ringing loudly.
Report the accident to 911.
We need a rescue team now.
The ambulance is on its way.
A tree has fallen on the road.
There's a gas leak in the kitchen.
Someone fainted on the street.
A fight broke out at the bar.
Send a paramedic to this location.
My car was stolen; call the police.
There's a strange smell in the house.
The fire truck is arriving soon.
A man is holding a weapon.
The baby is choking; get help!
The river is flooding the streets.
Call for backup immediately.
We're stuck in an elevator.
The building might collapse soon.
There's a bomb threat at the mall.
Someone fell down the stairs.
The burglar alarm is going off.
A motorcycle just caught fire.
Call animal control for the wild dog.
I can hear someone screaming.
There's a medical emergency here.
The ceiling is about to cave in.
The sirens are getting louder.
A car is stuck on the train tracks.
The lifeguard needs assistance.
A bicycle hit a pedestrian.
There's a suspicious package at the station.
A tree branch hit the power line.
Lightning struck the house nearby.
Someone is bleeding heavily.
The fire is spreading quickly.
I lost my wallet; call the police.
The ambulance just arrived.
The police officer is directing traffic.
There's been a shooting incident.
Someone is trying to break in.
Call the coast guard immediately.
A tanker truck overturned on the highway.
The emergency exits are blocked.
The firefighter rescued the cat.
Someone is having a heart attack.
Smoke is coming from the forest.
The child is locked in the car.
A cyclist was hit by a car.
I need the fire department here now.
The rescue boat is approaching.
Call for an air ambulance.
There's a power outage in the area.
The crowd is getting out of control.
A car exploded in the parking lot.
The patient needs oxygen immediately.
The storm caused severe damage.
There's an injured animal on the road.
The suspect is running away.
Someone reported a missing child.
The fire department is investigating.
Call 911 for emergency services.
The lifeguard saved the swimmer.
A pipe burst, and water is flooding.
There's a collapsed bridge ahead.
A truck is blocking the highway.
Call the hazmat team for the spill.
The security alarm won't stop.
The suspect is armed and dangerous.
A fire broke out at the school.
The ambulance is stuck in traffic.
The road is slippery due to the oil spill.
Someone fell into the open manhole.
The power lines are sparking.
The riverbank is overflowing.
The plane had to make an emergency landing.
Call the paramedics for this injury.
The fire engine is coming down the street.
The rescue helicopter is hovering above.
The building is on lockdown.
A landslide has blocked the road.
The bridge might collapse soon.
There's a riot breaking out downtown.
The sirens are heading toward the hospital.
Call the emergency number for help.
The firefighter climbed the ladder.
A gas explosion occurred at the factory.
The first responders are arriving now.
'''


if __name__ == "__main__":
  data = []
  labels = LABELS.split("\n")
  count = 1
  for lbl in labels:
    if (len(lbl) != 0):
      temp_data = {}
      temp_data['audio'] = f"data/custom/Recording ({count}).wav"
      temp_data['text'] = lbl
      data.append(temp_data)
      count += 1

  df = pd.DataFrame(data)
  df.to_csv("data_auto.csv", index=False)