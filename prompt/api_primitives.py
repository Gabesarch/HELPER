# You will be asked to generate Python code to follow navigation and manipulation instructions following the API below.

class InteractionObject:
    """
    This class represents an object in the house.
    """
    def __init__(self, object_class: str, landmark: str = None, attributes: list = []):
        '''
        object_class: object category of the interaction object (e.g., "Mug", "Apple")
        landmark: (optional if mentioned) landmark object category that the interaction object is in relation to (e.g., "CounterTop" for "apple is on the countertop")
        attributes: (optional) list of strings of desired attributes for the object. These are not necessarily attributes that currently exist, but ones that the object should eventually have. Attributes can only be from the following: "toasted", "clean", "cooked"
        '''
        self.object_class = object_class
        self.landmark = landmark
        self.attributes = attributes

    def pickup(self):
        """pickup the object.

        This function assumes the object is in view.

        Example:
        dialogue: <Commander> Go get the lettuce on the kitchen counter.
        Python script:
        target_lettuce = InteractionObject("Lettuce", landmark = "CounterTop")
        target_lettuce.go_to()
        target_lettuce.pickup()
        """
        pass

    def place(self, landmark_name):
        """put the interaction object on the landmark_name object.

        landmark_name must be a class InteractionObject instance

        This function assumes the robot has picked up an object and the landmark object is in view.

        Example:
        dialogue: <Commander> Put the lettuce on the kitchen counter.
        Python script:
        target_lettuce = InteractionObject("Lettuce", landmark = "CounterTop")
        target_lettuce.go_to()
        target_lettuce.pickup()
        target_countertop = InteractionObject("CounterTop")
        target_countertop.go_to()
        target_lettuce.place(target_countertop)
        """
        pass

    def slice(self):
        """slice the object into pieces.

        This function assumes the agent is holding a knife and the agent has navigated to the object using go_to().

        Example:
        dialogue: <Commander> Cut the apple on the kitchen counter.
        Python script:
        target_knife = InteractionObject("Knife") # first we need a knife to slice the apple with
        target_knife.go_to()
        target_knife.pickup()
        target_apple = InteractionObject("Apple", landmark = "CounterTop")
        target_apple.go_to()
        target_apple.slice()
        """
        pass

    def toggle_on(self):
        """toggles on the interaction object.

        This function assumes the interaction object is already off and the agent has navigated to the object.
        Only some landmark objects can be toggled on. Lamps, stoves, and microwaves are some examples of objects that can be toggled on.

        Example:
        dialogue: <Commander> Turn on the lamp.
        Python script:
        target_floorlamp = InteractionObject("FloorLamp")
        target_floorlamp.go_to()
        target_floorlamp.toggle_on()
        """
        pass

    def toggle_off(self):
        """toggles off the interaction object.

        This function assumes the interaction object is already on and the agent has navigated to the object.
        Only some objects can be toggled off. Lamps, stoves, and microwaves are some examples of objects that can be toggled off.

        Example:
        dialogue: <Commander> Turn off the lamp.
        Python script:
        target_floorlamp = InteractionObject("FloorLamp")
        target_floorlamp.go_to()
        target_floorlamp.toggle_off()
        """
        pass

    def go_to(self):
        """Navigate to the object

        """
        pass

    def open(self):
        """open the interaction object.

        This function assumes the landmark object is already closed and the agent has already navigated to the object.
        Only some objects can be opened. Fridges, cabinets, and drawers are some example of objects that can be closed.

        Example:
        dialogue: <Commander> Get the lettuce in the fridge.
        Python script:
        target_fridge = InteractionObject("Fridge")
        target_lettuce = InteractionObject("Lettuce", landmark = "Fridge")
        target_fridge.go_to()
        target_fridge.open()
        target_lettuce.pickup()
        """
        pass

    def close(self):
        """close the interaction object.

        This function assumes the object is already open and the agent has already navigated to the object.
        Only some objects can be closed. Fridges, cabinets, and drawers are some example of objects that can be closed.
        """
        pass
    
    def clean(self):
        """wash the interaction object to clean it in the sink.

        This function assumes the object is already picked up.

        Example:
        dialogue: <Commander> Clean the bowl
        Python script:
        target_bowl = InteractionObject("Bowl", attributes = ["clean"])
        target_bowl.clean()
        """
        pass

    def put_down(self):
        """puts the interaction object currently in the agent's hand on the nearest available receptacle

        This function assumes the object is already picked up.
        This function is most often used when the holding object is no longer needed, and the agent needs to pick up another object
        """
        pass

    def pour(self, landmark_name):
        """pours the contents of the interaction object into the landmark object specified by the landmark_name argument

        landmark_name must be a class InteractionObject instance

        This function assumes the object is already picked up and the object is filled with liquid. 
        """
        pass

    def fill_up(self):
        """fill up the interaction object with water

        This function assumes the object is already picked up. Note that only container objects can be filled with liquid. 
        """
        pass

    def pickup_and_place(self, landmark_name):
        """go_to() and pickup() this interaction object, then go_to() and place() the interaction object on the landmark_name object.

        landmark_name must be a class InteractionObject instance
        """
        pass

    def empty(self):
        """Empty the object of any other objects on/in it to clear it out. 

        Useful when the object is too full to place an object inside it.

        Example:
        dialogue: <Commander> Clear out the sink. 
        Python script:
        target_sink = InteractionObject("Sink")
        target_sink.empty()
        """
        pass

    def cook(self):
        """Cook the object

        Example:
        dialogue: <Commander> Cook the potato. 
        Python script:
        target_potato = InteractionObject("Potato", attributes = ["cooked"])
        target_potato.cook()
        """
        pass

    def toast(self):
        """Toast a bread slice in a toaster

        Toasting is only supported with slices of bread

        Example:
        dialogue: <Commander> Get me a toasted bread slice. 
        Python script:
        target_breadslice = InteractionObject("BreadSliced", attributes = ["toasted"])
        target_breadslice.toast()
        """
        pass