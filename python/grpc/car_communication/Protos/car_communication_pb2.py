# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: car_communication.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x17\x63\x61r_communication.proto\x12\x13\x43\x61rCommunicationApp\"\xb0\x01\n\x0bSensorsData\x12\x1b\n\x13\x66ront_left_distance\x18\x01 \x01(\x02\x12\x16\n\x0e\x66ront_distance\x18\x02 \x01(\x02\x12\x1c\n\x14\x66ront_right_distance\x18\x03 \x01(\x02\x12\x1a\n\x12\x62\x61\x63k_left_distance\x18\x04 \x01(\x02\x12\x15\n\rback_distance\x18\x05 \x01(\x02\x12\x1b\n\x13\x62\x61\x63k_right_distance\x18\x06 \x01(\x02\"\x82\x01\n\x07\x43ommand\x12\x39\n\tdirection\x18\x01 \x01(\x0e\x32&.CarCommunicationApp.Command.Direction\"<\n\tDirection\x12\x08\n\x04STOP\x10\x00\x12\x08\n\x04LEFT\x10\x01\x12\t\n\x05RIGHT\x10\x02\x12\x06\n\x02UP\x10\x03\x12\x08\n\x04\x44OWN\x10\x04\"\\\n\rClientRequest\x12\x13\n\x0bvideo_frame\x18\x01 \x01(\x0c\x12\x36\n\x0csensors_data\x18\x02 \x01(\x0b\x32 .CarCommunicationApp.SensorsData\"?\n\x0eServerResponse\x12-\n\x07\x63ommand\x18\x01 \x01(\x0b\x32\x1c.CarCommunicationApp.Command2g\n\rCommunication\x12V\n\x0bSendRequest\x12\".CarCommunicationApp.ClientRequest\x1a#.CarCommunicationApp.ServerResponseb\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'car_communication_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:
  DESCRIPTOR._options = None
  _globals['_SENSORSDATA']._serialized_start=49
  _globals['_SENSORSDATA']._serialized_end=225
  _globals['_COMMAND']._serialized_start=228
  _globals['_COMMAND']._serialized_end=358
  _globals['_COMMAND_DIRECTION']._serialized_start=298
  _globals['_COMMAND_DIRECTION']._serialized_end=358
  _globals['_CLIENTREQUEST']._serialized_start=360
  _globals['_CLIENTREQUEST']._serialized_end=452
  _globals['_SERVERRESPONSE']._serialized_start=454
  _globals['_SERVERRESPONSE']._serialized_end=517
  _globals['_COMMUNICATION']._serialized_start=519
  _globals['_COMMUNICATION']._serialized_end=622
# @@protoc_insertion_point(module_scope)