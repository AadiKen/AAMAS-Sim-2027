from bcod_sim.communication import CommunicationConfig, CommunicatingAction, MessageChannel
from bcod_sim.core.lifecycle import DirectAction
import pytest


def test_message_latency_range_and_reset():
    channel=MessageChannel(CommunicationConfig(True,2,1,5.,0.),7,("a","b","c"))
    positions={"a":(0.,0.),"b":(3.,0.),"c":(10.,0.)}
    channel.send(0,positions,{"a":(1.,-.5)})
    assert channel.receive(0,"b")["present"]==(False,False)
    delivered=channel.receive(1,"b")
    assert delivered["present"]==(True,False) and delivered["values"][0]==(1.,-.5)
    assert channel.receive(1,"c")["present"]==(False,False)
    channel.send(2,positions,{"a":(0.,0.)});channel.reset(7)
    assert channel.receive(3,"b")["present"]==(False,False)


def test_dropout_is_seeded_and_action_is_immutable_and_bounded():
    def trace():
        ch=MessageChannel(CommunicationConfig(True,1,1,None,.5),12,("a","b"))
        result=[]
        for step in range(12):
            ch.send(step,{"a":(0.,0.),"b":(1.,0.)},{"a":(.25,)})
            result.append(ch.receive(step+1,"b")["present"])
        return result
    assert trace()==trace()
    with pytest.raises(Exception): CommunicatingAction(DirectAction(()),(1.1,))
