import 'package:cloud_firestore/cloud_firestore.dart';

class BookingModel {
  final String id;
  final String studentId;
  final String mentorId;
  final String studentName;
  final String mentorName;
  final String? mentorPhoto;
  final String? studentPhoto;
  final DateTime date;
  final String timeSlot;
  final String topic;
  final String message;
  final String status; // pending | accepted | rejected | completed
  final DateTime createdAt;

  const BookingModel({
    required this.id,
    required this.studentId,
    required this.mentorId,
    required this.studentName,
    required this.mentorName,
    this.mentorPhoto,
    this.studentPhoto,
    required this.date,
    required this.timeSlot,
    required this.topic,
    required this.message,
    required this.status,
    required this.createdAt,
  });

  factory BookingModel.fromMap(Map<String, dynamic> map, String docId) {
    return BookingModel(
      id: docId,
      studentId: map['studentId'] as String? ?? '',
      mentorId: map['mentorId'] as String? ?? '',
      studentName: map['studentName'] as String? ?? '',
      mentorName: map['mentorName'] as String? ?? '',
      mentorPhoto: map['mentorPhoto'] as String?,
      studentPhoto: map['studentPhoto'] as String?,
      date: map['date'] is Timestamp
          ? (map['date'] as Timestamp).toDate()
          : DateTime.now(),
      timeSlot: map['timeSlot'] as String? ?? '',
      topic: map['topic'] as String? ?? '',
      message: map['message'] as String? ?? '',
      status: map['status'] as String? ?? 'pending',
      createdAt: map['createdAt'] is Timestamp
          ? (map['createdAt'] as Timestamp).toDate()
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() => {
        'studentId': studentId,
        'mentorId': mentorId,
        'studentName': studentName,
        'mentorName': mentorName,
        'mentorPhoto': mentorPhoto,
        'studentPhoto': studentPhoto,
        'date': Timestamp.fromDate(date),
        'timeSlot': timeSlot,
        'topic': topic,
        'message': message,
        'status': status,
        'createdAt': Timestamp.fromDate(createdAt),
      };

  BookingModel copyWith({String? status}) => BookingModel(
        id: id,
        studentId: studentId,
        mentorId: mentorId,
        studentName: studentName,
        mentorName: mentorName,
        mentorPhoto: mentorPhoto,
        studentPhoto: studentPhoto,
        date: date,
        timeSlot: timeSlot,
        topic: topic,
        message: message,
        status: status ?? this.status,
        createdAt: createdAt,
      );

  String get statusLabel {
    switch (status) {
      case 'pending':
        return 'Күтуде';
      case 'accepted':
        return 'Қабылданды';
      case 'rejected':
        return 'Қабылданбады';
      case 'completed':
        return 'Аяқталды';
      default:
        return 'Белгісіз';
    }
  }
}
