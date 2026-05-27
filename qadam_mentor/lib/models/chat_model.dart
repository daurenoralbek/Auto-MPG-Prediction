import 'package:cloud_firestore/cloud_firestore.dart';

class ChatModel {
  final String id;
  final List<String> participants;
  final String studentId;
  final String mentorId;
  final String studentName;
  final String mentorName;
  final String? studentPhoto;
  final String? mentorPhoto;
  final String lastMessage;
  final DateTime lastMessageTime;
  final int unreadCount;

  const ChatModel({
    required this.id,
    required this.participants,
    required this.studentId,
    required this.mentorId,
    required this.studentName,
    required this.mentorName,
    this.studentPhoto,
    this.mentorPhoto,
    required this.lastMessage,
    required this.lastMessageTime,
    required this.unreadCount,
  });

  factory ChatModel.fromMap(Map<String, dynamic> map, String docId) {
    return ChatModel(
      id: docId,
      participants: List<String>.from(map['participants'] as List? ?? []),
      studentId: map['studentId'] as String? ?? '',
      mentorId: map['mentorId'] as String? ?? '',
      studentName: map['studentName'] as String? ?? '',
      mentorName: map['mentorName'] as String? ?? '',
      studentPhoto: map['studentPhoto'] as String?,
      mentorPhoto: map['mentorPhoto'] as String?,
      lastMessage: map['lastMessage'] as String? ?? '',
      lastMessageTime: map['lastMessageTime'] is Timestamp
          ? (map['lastMessageTime'] as Timestamp).toDate()
          : DateTime.now(),
      unreadCount: map['unreadCount'] as int? ?? 0,
    );
  }

  Map<String, dynamic> toMap() => {
        'participants': participants,
        'studentId': studentId,
        'mentorId': mentorId,
        'studentName': studentName,
        'mentorName': mentorName,
        'studentPhoto': studentPhoto,
        'mentorPhoto': mentorPhoto,
        'lastMessage': lastMessage,
        'lastMessageTime': Timestamp.fromDate(lastMessageTime),
        'unreadCount': unreadCount,
      };

  String otherPersonName(String currentUid) =>
      currentUid == studentId ? mentorName : studentName;

  String? otherPersonPhoto(String currentUid) =>
      currentUid == studentId ? mentorPhoto : studentPhoto;
}
