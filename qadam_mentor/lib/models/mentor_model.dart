import 'package:cloud_firestore/cloud_firestore.dart';

class MentorModel {
  final String id;
  final String uid;
  final String name;
  final String email;
  final String? photoUrl;
  final String bio;
  final String specialty;
  final List<String> categories;
  final double rating;
  final int reviewCount;
  final bool isAvailable;
  final List<String> availableDays;
  final String availableHours;
  final String role; // 'teacher' | 'senior_student' | 'club_leader'
  final DateTime createdAt;

  const MentorModel({
    required this.id,
    required this.uid,
    required this.name,
    required this.email,
    this.photoUrl,
    required this.bio,
    required this.specialty,
    required this.categories,
    required this.rating,
    required this.reviewCount,
    required this.isAvailable,
    required this.availableDays,
    required this.availableHours,
    required this.role,
    required this.createdAt,
  });

  factory MentorModel.fromMap(Map<String, dynamic> map, String docId) {
    return MentorModel(
      id: docId,
      uid: map['uid'] as String? ?? docId,
      name: map['name'] as String? ?? '',
      email: map['email'] as String? ?? '',
      photoUrl: map['photoUrl'] as String?,
      bio: map['bio'] as String? ?? '',
      specialty: map['specialty'] as String? ?? '',
      categories: List<String>.from(map['categories'] as List? ?? []),
      rating: (map['rating'] as num?)?.toDouble() ?? 0.0,
      reviewCount: map['reviewCount'] as int? ?? 0,
      isAvailable: map['isAvailable'] as bool? ?? true,
      availableDays: List<String>.from(map['availableDays'] as List? ?? []),
      availableHours: map['availableHours'] as String? ?? '09:00–17:00',
      role: map['role'] as String? ?? 'teacher',
      createdAt: map['createdAt'] is Timestamp
          ? (map['createdAt'] as Timestamp).toDate()
          : DateTime.now(),
    );
  }

  Map<String, dynamic> toMap() => {
        'uid': uid,
        'name': name,
        'email': email,
        'photoUrl': photoUrl,
        'bio': bio,
        'specialty': specialty,
        'categories': categories,
        'rating': rating,
        'reviewCount': reviewCount,
        'isAvailable': isAvailable,
        'availableDays': availableDays,
        'availableHours': availableHours,
        'role': role,
        'createdAt': Timestamp.fromDate(createdAt),
      };

  String get roleLabel {
    switch (role) {
      case 'teacher':
        return 'Мұғалім';
      case 'senior_student':
        return 'Ағалық студент';
      case 'club_leader':
        return 'Клуб жетекшісі';
      default:
        return 'Ментор';
    }
  }
}
